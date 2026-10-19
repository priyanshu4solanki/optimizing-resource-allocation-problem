"""Train and evaluate the reference-style MAPPO scheduler on the local pipeline."""

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from main_engine import (
    Server,
    _demo_workload,
    _estimate_execution_time,
    _fits,
    _server_load,
    _server_telemetry,
    _topological_order,
)
from mappo_scheduler import (
    MAPPORolloutTransition,
    MAPPOScheduler,
)
from marl_scheduler_bridge import build_joint_state
from model_persistence import load_gnn_encoder, load_predictor
from gnn_encoder import build_dependency_edges, task_gnn_feature_tensor

REFERENCE_REPOSITORY = "https://github.com/YashSigchi/Capstone_MARL"
REFERENCE_COMMIT = "681666a81ca24c7c1c8067744f3ac56116569659"
TELEMETRY_DIM = 7
REWARD_WEIGHTS = {
    "action_penalty": 0.10,
    "admission": 0.10,
    "energy": 0.15,
    "latency": 0.15,
    "load_balance": 0.10,
    "queue": 0.10,
    "sla": 0.10,
    "throughput": 0.05,
    "utilization": 0.15,
}
REWARD_BLEND_ALPHA = 0.5
REJECTION_TARGET = 0.20


def _servers() -> Tuple[Server, ...]:
    return (
        Server("edge-small", 4, 1, 16, 100, 18, 0.015, initial_load=0.2),
        Server("cloud-balanced", 8, 4, 32, 500, 35, 0.035, initial_load=0.1),
        Server("cloud-fast", 16, 8, 64, 900, 48, 0.08, initial_load=0.05),
    )


def _make_scheduler(embedding_dim: int, task_count: int, seed: int) -> MAPPOScheduler:
    bridge_input_dim = embedding_dim + TELEMETRY_DIM
    observation_dim = len(_servers()) * bridge_input_dim
    global_state_dim = task_count * (observation_dim + 4) + 8
    torch.manual_seed(seed)
    return MAPPOScheduler(
        input_dim=bridge_input_dim,
        num_servers=len(_servers()),
        observation_dim=observation_dim,
        global_state_dim=global_state_dim,
        actor_hidden_dim=128,
        critic_hidden_dim=256,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.001,
        update_epochs=4,
        minibatches=4,
    )


class _SchedulingEpisode:
    def __init__(
        self,
        predictor,
        encoder,
        tasks,
        contexts,
        servers: Sequence[Server],
        *,
        seed: int,
    ) -> None:
        if len(tasks) != len(contexts):
            raise ValueError("tasks and contexts must have equal lengths")
        if len(tasks) == 0:
            raise ValueError("an episode needs at least one task")
        self.tasks = tuple(tasks)
        self.servers = tuple(servers)
        self.rng = np.random.default_rng(seed)
        self.predictions, self.embeddings = self._predict(
            predictor, encoder, tasks, contexts
        )
        self.task_ids = tuple(task.task_id for task in tasks)
        parents = {task.task_id: task.parents for task in tasks}
        self.order = _topological_order(self.task_ids, parents)
        self.index_by_id = {task_id: index for index, task_id in enumerate(self.task_ids)}
        self.max_cost = max(server.cost_per_second for server in servers)
        self.remaining = {
            server.server_id: {
                "cpu": server.cpu_capacity * (1 - server.initial_load),
                "gpu": server.gpu_capacity * (1 - server.initial_load),
                "ram": server.ram_capacity * (1 - server.initial_load),
            }
            for server in servers
        }
        self.deadlines = {
            task_id: self.predictions[task_id].target_exec_time
            * float(self.rng.uniform(1.2, 1.8))
            for task_id in self.task_ids
        }
        self.priorities = {
            task_id: int(self.rng.integers(1, 6)) for task_id in self.task_ids
        }
        self.status = {task_id: 0 for task_id in self.task_ids}
        self.deferred = set()
        self.queue = list(self.order)
        self.rejected = 0
        self.completed = 0
        self.schedules: List[Dict[str, object]] = []
        self.components: List[Dict[str, float]] = []

    @staticmethod
    def _predict(predictor, encoder, tasks, contexts):
        predictions = predictor.predict(tasks, contexts)
        task_ids = tuple(task.task_id for task in tasks)
        parents = {task.task_id: task.parents for task in tasks}
        edges = build_dependency_edges(task_ids, parents)
        features = task_gnn_feature_tensor(tasks, contexts, predictions)
        device = next(encoder.parameters()).device
        refined, embeddings = encoder.predict_resources(
            features.to(device), edges.to(device)
        )
        updated = {
            prediction.task_id: replace(
                prediction,
                target_exec_time=float(resource[0]),
                target_cpu_req=max(prediction.target_cpu_req, float(resource[1])),
                target_gpu_req=max(prediction.target_gpu_req, float(resource[2])),
                target_ram_req=float(resource[3]),
            )
            for prediction, resource in zip(predictions, refined.cpu())
        }
        return updated, embeddings.detach().cpu().to(torch.float32)

    def _telemetry(self) -> torch.Tensor:
        return torch.tensor(
            [
                _server_telemetry(
                    server, self.remaining[server.server_id], self.max_cost
                )
                for server in self.servers
            ],
            dtype=torch.float32,
        )

    def _observation(self, task_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        index = self.index_by_id[task_id]
        bridge = build_joint_state(
            self.embeddings[index].unsqueeze(0),
            self._telemetry(),
            task_ids=(task_id,),
            server_ids=tuple(server.server_id for server in self.servers),
            feasibility_mask=torch.tensor(
                [
                    [
                        _fits(
                            self.predictions[task_id],
                            self.remaining[server.server_id],
                        )
                        for server in self.servers
                    ]
                ],
                dtype=torch.bool,
            ),
        )
        return bridge.joint_state.flatten(), bridge.feasibility_mask.flatten()

    def _global_state(self, step: int) -> torch.Tensor:
        telemetry = self._telemetry()
        all_tasks = []
        for task_id in self.task_ids:
            observation, _ = self._observation(task_id)
            task_state = torch.cat(
                (
                    observation,
                    torch.nn.functional.one_hot(
                        torch.tensor(self.status[task_id]), num_classes=4
                    ).to(torch.float32),
                )
            )
            all_tasks.append(task_state)
        loads = torch.tensor(
            [
                _server_load(server, self.remaining[server.server_id])
                for server in self.servers
            ],
            dtype=torch.float32,
        )
        total = len(self.task_ids)
        summary = torch.tensor(
            [
                loads.mean().item(),
                loads.std(unbiased=False).item(),
                loads.max().item(),
                loads.min().item(),
                self.rejected / total,
                self.completed / total,
                sum(status in (0, 1) for status in self.status.values()) / total,
                min(step / max(total, 1), 1.0),
            ],
            dtype=torch.float32,
        )
        return torch.cat((*all_tasks, summary))

    def _action_mask(self, task_id: str, feasible: torch.Tensor) -> torch.Tensor:
        defer_valid = task_id not in self.deferred
        reject_limit = max(1, math.floor(len(self.task_ids) * REJECTION_TARGET))
        reject_valid = self.rejected < reject_limit or not feasible.any()
        return torch.cat(
            (
                feasible,
                torch.tensor([defer_valid, reject_valid], dtype=torch.bool),
            )
        )

    def _reward_components(
        self,
        *,
        action: int,
        task_id: str,
        feasible: torch.Tensor,
        execution_time: float = 0.0,
        cost: float = 0.0,
        placed: bool = False,
        rejected: bool = False,
    ) -> Dict[str, float]:
        total = len(self.task_ids)
        loads = [
            _server_load(server, self.remaining[server.server_id])
            for server in self.servers
        ]
        mean_load = sum(loads) / len(loads)
        if mean_load < 0.4:
            utilization_reward = mean_load / 0.4
        elif mean_load <= 0.8:
            utilization_reward = 1.0
        else:
            utilization_reward = max(0.0, (1.0 - mean_load) / 0.2)
        deadline = max(self.deadlines[task_id], 1e-6)
        if placed:
            latency = max(-1.0, min(1.0, 1.0 - execution_time / deadline))
            sla = (
                min(1.0, max(0.0, (deadline - execution_time) / deadline))
                * self.priorities[task_id]
                / 5.0
                if execution_time <= deadline
                else -self.priorities[task_id] / 5.0
            )
        elif rejected:
            latency = 0.0
            sla = -self.priorities[task_id] / 5.0
        else:
            latency = 0.0
            sla = 0.0
        action_penalty = (
            -1.0
            if action == len(self.servers) and feasible.any()
            else -0.1
            if action >= len(self.servers)
            else 0.0
        )
        rejection_rate = self.rejected / total
        admission = -min(
            5.0,
            rejection_rate * 5.0
            + max(0.0, rejection_rate - REJECTION_TARGET) * 10.0,
        )
        pending = sum(status in (0, 1) for status in self.status.values())
        return {
            "action_penalty": action_penalty,
            "admission": admission,
            "energy": -min(1.0, max(0.0, cost * 1000.0 / 10000.0)),
            "latency": latency,
            "load_balance": -min(
                1.0,
                sum((load - mean_load) ** 2 for load in loads) / len(loads) * 10.0,
            ),
            "queue": -min(1.0, pending / total),
            "sla": sla,
            "throughput": 1.0 if placed else 0.0,
            "utilization": utilization_reward,
        }

    @staticmethod
    def _combine_reward(components: Dict[str, float]) -> float:
        local_names = ("latency", "energy", "sla", "queue", "action_penalty")
        global_names = (
            "utilization",
            "load_balance",
            "throughput",
            "admission",
        )
        local_reward = sum(REWARD_WEIGHTS[name] * components[name] for name in local_names)
        global_reward = sum(
            REWARD_WEIGHTS[name] * components[name] for name in global_names
        )
        return (
            REWARD_BLEND_ALPHA * local_reward
            + (1 - REWARD_BLEND_ALPHA) * global_reward
        )

    def run(self, scheduler: MAPPOScheduler, *, policy: str) -> Dict[str, object]:
        transitions: List[MAPPORolloutTransition] = []
        episode_reward = 0.0
        step = 0
        while self.queue:
            task_id = self.queue.pop(0)
            observation, feasible = self._observation(task_id)
            action_mask = self._action_mask(task_id, feasible)
            global_state = self._global_state(step)
            if policy in ("mappo", "train"):
                chosen = scheduler.act(
                    observation,
                    global_state,
                    action_mask,
                    deterministic=(policy == "mappo"),
                )
                action = chosen.action
            elif policy == "random":
                options = torch.where(action_mask)[0].numpy()
                action = int(self.rng.choice(options))
                chosen = None
            elif policy == "cheapest":
                valid = torch.where(feasible)[0].tolist()
                action = min(valid, key=lambda index: self.servers[index].cost_per_second)
                chosen = None
            elif policy == "fastest":
                valid = torch.where(feasible)[0].tolist()
                prediction = self.predictions[task_id]
                action = min(
                    valid,
                    key=lambda index: _estimate_execution_time(
                        prediction, self.servers[index]
                    ),
                )
                chosen = None
            else:
                raise ValueError(f"unknown policy {policy!r}")

            server_count = len(self.servers)
            placed = False
            rejected = False
            simulated_time = 0.0
            simulated_cost = 0.0
            if action < server_count:
                if not feasible[action]:
                    raise RuntimeError("policy selected an infeasible server")
                server = self.servers[action]
                prediction = self.predictions[task_id]
                simulated_time = _estimate_execution_time(prediction, server)
                simulated_cost = simulated_time * server.cost_per_second
                for resource in ("cpu", "gpu", "ram"):
                    self.remaining[server.server_id][resource] -= getattr(
                        prediction,
                        {
                            "cpu": "target_cpu_req",
                            "gpu": "target_gpu_req",
                            "ram": "target_ram_req",
                        }[resource],
                    )
                self.status[task_id] = 2
                self.completed += 1
                placed = True
                self.schedules.append(
                    {
                        "task_id": task_id,
                        "server_id": server.server_id,
                        "execution_time": simulated_time,
                        "cost": simulated_cost,
                        "deadline": self.deadlines[task_id],
                        "priority": self.priorities[task_id],
                    }
                )
            elif action == server_count:
                self.status[task_id] = 1
                self.deferred.add(task_id)
                self.queue.append(task_id)
            else:
                self.status[task_id] = 3
                self.rejected += 1
                rejected = True

            components = self._reward_components(
                action=action,
                task_id=task_id,
                feasible=feasible,
                execution_time=simulated_time,
                cost=simulated_cost,
                placed=placed,
                rejected=rejected,
            )
            reward = self._combine_reward(components)
            episode_reward += reward
            self.components.append(components)
            if chosen is not None:
                transitions.append(
                    MAPPORolloutTransition(
                        observation=chosen.observation,
                        global_state=chosen.global_state,
                        action_mask=chosen.action_mask,
                        action=chosen.action,
                        log_probability=chosen.log_probability,
                        value=chosen.value,
                        reward=reward,
                        done=not self.queue,
                    )
                )
            step += 1

        if transitions and not transitions[-1].done:
            transitions[-1] = replace(transitions[-1], done=True)
        if self.completed + self.rejected != len(self.task_ids):
            raise RuntimeError("episode finished with unresolved tasks")
        loads = [
            _server_load(server, self.remaining[server.server_id])
            for server in self.servers
        ]
        times = [float(item["execution_time"]) for item in self.schedules]
        return {
            "episode_reward": episode_reward,
            "mean_step_reward": episode_reward / max(step, 1),
            "completed": self.completed,
            "rejected": self.rejected,
            "deferred_decisions": len(self.deferred),
            "sla_violations": sum(
                float(item["execution_time"]) > float(item["deadline"])
                for item in self.schedules
            ),
            "mean_latency": float(np.mean(times)) if times else 0.0,
            "energy_proxy": sum(float(item["cost"]) * 1000 for item in self.schedules),
            "mean_utilization": float(np.mean(loads)),
            "load_variance": float(np.var(loads)),
            "completion_rate": self.completed / len(self.task_ids),
            "rejection_rate": self.rejected / len(self.task_ids),
            "sla_violation_rate": (
                sum(
                    float(item["execution_time"]) > float(item["deadline"])
                    for item in self.schedules
                )
                / max(self.completed, 1)
            ),
            "defer_rate": len(self.deferred) / len(self.task_ids),
            "steps": step,
            "transitions": transitions,
            "reward_components": {
                name: float(np.mean([row[name] for row in self.components]))
                for name in REWARD_WEIGHTS
            },
        }


def _episode(
    predictor,
    encoder,
    task_count: int,
    seed: int,
    scheduler: MAPPOScheduler,
    policy: str,
) -> Dict[str, object]:
    tasks, contexts = _demo_workload(task_count, seed)
    environment = _SchedulingEpisode(
        predictor, encoder, tasks, contexts, _servers(), seed=seed + 991
    )
    return environment.run(scheduler, policy=policy)


def _evaluate(
    predictor,
    encoder,
    scheduler: MAPPOScheduler,
    *,
    task_count: int,
    seeds: Sequence[int],
    policy: str,
) -> Dict[str, object]:
    rows = [
        _episode(predictor, encoder, task_count, seed, scheduler, policy)
        for seed in seeds
    ]
    metrics = (
        "episode_reward",
        "completion_rate",
        "rejection_rate",
        "sla_violation_rate",
        "mean_latency",
        "energy_proxy",
        "mean_utilization",
        "load_variance",
        "defer_rate",
    )
    return {
        "policy": policy,
        "seeds": list(seeds),
        "episodes": len(rows),
        "mean": {name: float(np.mean([row[name] for row in rows])) for name in metrics},
        "std": {name: float(np.std([row[name] for row in rows])) for name in metrics},
        "per_seed": [
            {key: value for key, value in row.items() if key != "transitions"}
            for row in rows
        ],
    }


def train(
    *,
    episodes: int,
    task_count: int,
    eval_seeds: int,
    seed: int,
    model_dir: Path,
    output_dir: Path,
) -> Dict[str, object]:
    if episodes <= 0 or task_count <= 0 or eval_seeds <= 0:
        raise ValueError("episodes, tasks, and eval-seeds must all be positive")
    predictor_path = model_dir / "task_predictor.joblib"
    encoder_path = model_dir / "task_gnn.pt"
    if not predictor_path.is_file() or not encoder_path.is_file():
        raise FileNotFoundError(
            "The fixed predictor and GNN checkpoints are required. Run "
            "main_engine.py --checkpoint-dir models before MAPPO training."
        )
    predictor = load_predictor(predictor_path)
    encoder = load_gnn_encoder(encoder_path)
    encoder.eval()
    scheduler = _make_scheduler(encoder.embedding_dim, task_count, seed)

    eval_set = [seed + 50000 + index for index in range(eval_seeds)]
    baseline_results = {
        policy: _evaluate(
            predictor,
            encoder,
            scheduler,
            task_count=task_count,
            seeds=eval_set,
            policy=policy,
        )
        for policy in ("random", "cheapest", "fastest")
    }
    untrained_result = _evaluate(
        predictor,
        encoder,
        scheduler,
        task_count=task_count,
        seeds=eval_set,
        policy="mappo",
    )

    history: List[Dict[str, object]] = []
    for episode_index in range(episodes):
        episode_seed = seed + 1000 + episode_index
        row = _episode(
            predictor,
            encoder,
            task_count,
            episode_seed,
            scheduler,
            "train",
        )
        update = scheduler.update(row["transitions"])
        history.append(
            {
                "episode": episode_index + 1,
                "seed": episode_seed,
                **{key: value for key, value in row.items() if key not in ("transitions", "reward_components")},
                **{f"reward_{key}": value for key, value in row["reward_components"].items()},
                **update,
            }
        )

    trained_result = _evaluate(
        predictor,
        encoder,
        scheduler,
        task_count=task_count,
        seeds=eval_set,
        policy="mappo",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    scheduler.save(str(output_dir / "mappo_scheduler.pt"))
    config = {
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "algorithm": "shared-actor centralized-critic MAPPO with GAE",
        "local_inputs": "existing GNN task embedding + existing 7-value server telemetry per server",
        "action_space": [
            *[f"place_on_server_{index}" for index in range(len(_servers()))],
            "defer_once",
            "reject",
        ],
        "actor_hidden_sizes": [128, 128],
        "critic_hidden_sizes": [256, 256],
        "gamma": scheduler.gamma,
        "gae_lambda": scheduler.gae_lambda,
        "clip_epsilon": scheduler.clip_epsilon,
        "actor_learning_rate": scheduler.learning_rate,
        "critic_learning_rate": scheduler.learning_rate,
        "entropy_coefficient": scheduler.entropy_coefficient,
        "value_coefficient": scheduler.value_coefficient,
        "gradient_norm_clip": 0.5,
        "ppo_epochs": scheduler.update_epochs,
        "minibatches": scheduler.minibatches,
        "reward_weights": REWARD_WEIGHTS,
        "reward_blend_alpha": REWARD_BLEND_ALPHA,
        "rejection_target": REJECTION_TARGET,
        "synthetic_deadline": "1.2-1.8 times GNN-refined execution estimate; reward-only",
        "energy_measurement": "cost * 1000 proxy because the local simulator has no power telemetry",
        "fixed_predictor_checkpoint": str(predictor_path),
        "fixed_gnn_checkpoint": str(encoder_path),
        "seed": seed,
        "training_episodes": episodes,
        "task_count_per_episode": task_count,
        "evaluation_seeds": eval_set,
        "server_count": len(_servers()),
    }
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    report = {
        "configuration": config,
        "untrained_policy": untrained_result,
        "baselines": baseline_results,
        "trained_policy": trained_result,
        "trained_vs_untrained_reward_delta": (
            trained_result["mean"]["episode_reward"]
            - untrained_result["mean"]["episode_reward"]
        ),
        "trained_vs_fastest_reward_delta": (
            trained_result["mean"]["episode_reward"]
            - baseline_results["fastest"]["mean"]["episode_reward"]
        ),
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _plot_training(history, output_dir / "training_curves.png")
    _plot_evaluation(report, output_dir / "evaluation_comparison.png")
    _write_run_log(history, report, output_dir / "training.log")
    return report


def _plot_training(history: Sequence[Dict[str, object]], path: Path) -> None:
    episodes = [int(row["episode"]) for row in history]
    rewards = [float(row["episode_reward"]) for row in history]
    window = min(20, len(rewards))
    smooth = np.convolve(rewards, np.ones(window) / window, mode="valid")
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="episode reward")
    plt.plot(episodes[window - 1 :], smooth, linewidth=2, label=f"{window}-episode mean")
    plt.xlabel("Training episode")
    plt.ylabel("Summed reward")
    plt.title("MAPPO training reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_evaluation(report: Dict[str, object], path: Path) -> None:
    policies = {
        "Untrained MAPPO": report["untrained_policy"],
        "Trained MAPPO": report["trained_policy"],
        "Random feasible": report["baselines"]["random"],
        "Cheapest": report["baselines"]["cheapest"],
        "Fastest": report["baselines"]["fastest"],
    }
    names = list(policies)
    reward_values = [
        policies[name]["mean"]["episode_reward"] for name in names
    ]
    completion_values = [
        policies[name]["mean"]["completion_rate"] for name in names
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(names, reward_values)
    axes[0].set_title("Mean held-out reward")
    axes[0].set_ylabel("Reward per episode")
    axes[1].bar(names, completion_values)
    axes[1].set_title("Mean held-out completion rate")
    axes[1].set_ylim(0, 1.05)
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_run_log(
    history: Sequence[Dict[str, object]], report: Dict[str, object], path: Path
) -> None:
    lines = [
        f"Training episodes: {len(history)}",
        f"Final training reward: {history[-1]['episode_reward']:.6f}",
        "Held-out evaluation means:",
    ]
    results = {
        "Untrained MAPPO": report["untrained_policy"],
        **{f"Baseline {key}": value for key, value in report["baselines"].items()},
        "Trained MAPPO": report["trained_policy"],
    }
    for label, evaluation in results.items():
        metrics = evaluation["mean"]
        lines.append(
            f"{label}: reward={metrics['episode_reward']:.6f}, "
            f"completion={metrics['completion_rate']:.4f}, "
            f"rejection={metrics['rejection_rate']:.4f}, "
            f"SLA_violations={metrics['sla_violation_rate']:.4f}, "
            f"latency={metrics['mean_latency']:.4f}, "
            f"energy_proxy={metrics['energy_proxy']:.4f}"
        )
    lines.append(
        "Trained-vs-untrained reward delta: "
        f"{report['trained_vs_untrained_reward_delta']:.6f}"
    )
    lines.append(
        "Trained-vs-fastest reward delta: "
        f"{report['trained_vs_fastest_reward_delta']:.6f}"
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--eval-seeds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results") / "mappo_reference"
    )
    args = parser.parse_args()
    report = train(
        episodes=args.episodes,
        task_count=args.tasks,
        eval_seeds=args.eval_seeds,
        seed=args.seed,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
    )
    print(f"Saved MAPPO checkpoint and results in: {args.output_dir.resolve()}")
    print(
        "Held-out mean reward: "
        f"untrained={report['untrained_policy']['mean']['episode_reward']:.4f}, "
        f"trained={report['trained_policy']['mean']['episode_reward']:.4f}, "
        f"fastest={report['baselines']['fastest']['mean']['episode_reward']:.4f}"
    )
    print(
        "Trained-vs-untrained reward delta: "
        f"{report['trained_vs_untrained_reward_delta']:.4f}"
    )


if __name__ == "__main__":
    main()
