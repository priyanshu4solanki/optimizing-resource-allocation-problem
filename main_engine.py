"""Run an offline task-ingestion to GNN to MARL scheduling simulation."""

import argparse
from dataclasses import dataclass
from dataclasses import replace
from math import isfinite
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from gnn_encoder import (
    TaskGraphEncoder,
    build_dependency_edges,
    task_gnn_feature_tensor,
)
from marl_scheduler import (
    AdaptiveRewardConfig,
    MARLScheduler,
    SchedulerDecision,
    SchedulerExperience,
)
from marl_scheduler_bridge import SchedulerBridgeState, build_joint_state
from model_persistence import save_gnn_encoder, save_predictor
from task_predictors import (
    InfrastructureContext,
    PredictedTask,
    TaskInput,
    TaskResourcePredictor,
    generate_synthetic_training_data,
    synthetic_resource_targets,
)


@dataclass(frozen=True)
class Server:
    server_id: str
    cpu_capacity: float
    gpu_capacity: float
    ram_capacity: float
    bandwidth: float
    latency: float
    cost_per_second: float
    initial_load: float = 0.0

    def __post_init__(self) -> None:
        if not self.server_id:
            raise ValueError("server_id must not be empty")
        for name in (
            "cpu_capacity",
            "gpu_capacity",
            "ram_capacity",
            "bandwidth",
            "cost_per_second",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.latency) or self.latency < 0:
            raise ValueError("latency must be finite and non-negative")
        if not isfinite(self.initial_load) or not 0 <= self.initial_load < 1:
            raise ValueError("initial_load must be in [0, 1)")


@dataclass(frozen=True)
class TaskSchedule:
    task_id: str
    server_id: str
    predicted_execution_time: float
    simulated_execution_time: float
    simulated_cost: float
    reward: float
    execution_time_weight: float
    cost_weight: float


@dataclass(frozen=True)
class SimulationResult:
    schedules: Tuple[TaskSchedule, ...]
    embeddings: torch.Tensor
    scheduler_update: Optional[dict]
    gnn_metrics: Optional[dict] = None


class OfflineSchedulingEngine:
    """Compose the predictor, task GNN, bridge, and scheduler in one loop."""

    def __init__(
        self,
        predictor: TaskResourcePredictor,
        encoder: TaskGraphEncoder,
        scheduler: MARLScheduler,
        servers: Sequence[Server],
    ) -> None:
        if not servers:
            raise ValueError("at least one server is required")
        if len({server.server_id for server in servers}) != len(servers):
            raise ValueError("server IDs must be unique")
        if scheduler.input_dim != encoder.embedding_dim + 7:
            raise ValueError(
                "scheduler input_dim must equal encoder.embedding_dim + 7 telemetry features"
            )
        if encoder.output_dim != 4:
            raise ValueError("encoder must predict the four task resource targets")
        self.predictor = predictor
        self.encoder = encoder
        self.scheduler = scheduler
        self.servers = tuple(servers)

    def run(
        self,
        tasks: Sequence[TaskInput],
        contexts: Sequence[InfrastructureContext],
        *,
        train_scheduler: bool = True,
        deterministic: bool = False,
    ) -> SimulationResult:
        if not tasks:
            raise ValueError("at least one task is required")
        if len(tasks) != len(contexts):
            raise ValueError("tasks and contexts must have the same length")
        task_ids = tuple(task.task_id for task in tasks)
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task IDs must be unique")
        predictions = self.predictor.predict(tasks, contexts)
        parents_by_task = {task.task_id: task.parents for task in tasks}
        edge_index = build_dependency_edges(task_ids, parents_by_task)
        features = task_gnn_feature_tensor(tasks, contexts, predictions)
        model_device = next(self.encoder.parameters()).device
        refined_features, encoded = self.encoder.predict_resources(
            features.to(model_device), edge_index.to(model_device)
        )
        embeddings = encoded.to(device="cpu", dtype=torch.float32)
        predictions = [
            replace(
                prediction,
                target_exec_time=float(refined[0]),
                target_cpu_req=max(
                    prediction.target_cpu_req, float(refined[1])
                ),
                target_gpu_req=max(
                    prediction.target_gpu_req, float(refined[2])
                ),
                target_ram_req=float(refined[3]),
            )
            for prediction, refined in zip(predictions, refined_features.cpu())
        ]
        order = _topological_order(task_ids, parents_by_task)
        prediction_by_id = {item.task_id: item for item in predictions}
        embedding_index = {task_id: index for index, task_id in enumerate(task_ids)}
        remaining = {
            server.server_id: {
                "cpu": server.cpu_capacity * (1 - server.initial_load),
                "gpu": server.gpu_capacity * (1 - server.initial_load),
                "ram": server.ram_capacity * (1 - server.initial_load),
            }
            for server in self.servers
        }
        max_cost = max(server.cost_per_second for server in self.servers)
        rollout: List[SchedulerExperience] = []
        schedules: List[TaskSchedule] = []

        for position, task_id in enumerate(order):
            prediction = prediction_by_id[task_id]
            telemetry = torch.tensor(
                [_server_telemetry(server, remaining[server.server_id], max_cost)
                 for server in self.servers],
                dtype=torch.float32,
            )
            feasible = torch.tensor(
                [
                    [
                        _fits(prediction, remaining[server.server_id])
                        for server in self.servers
                    ]
                ],
                dtype=torch.bool,
            )
            if not feasible.any():
                raise RuntimeError(
                    f"no server has sufficient remaining resources for task {task_id!r}"
                )
            bridge_state = build_joint_state(
                embeddings[embedding_index[task_id]].unsqueeze(0),
                telemetry,
                task_ids=(task_id,),
                server_ids=tuple(server.server_id for server in self.servers),
                feasibility_mask=feasible,
            )
            decision: SchedulerDecision = self.scheduler.select_actions(
                bridge_state, deterministic=deterministic
            )
            server = self.servers[decision.action_indices[0]]
            system_load = _server_load(server, remaining[server.server_id])
            remaining[server.server_id]["cpu"] -= prediction.target_cpu_req
            remaining[server.server_id]["gpu"] -= prediction.target_gpu_req
            remaining[server.server_id]["ram"] -= prediction.target_ram_req

            simulated_time = _estimate_execution_time(prediction, server)
            simulated_cost = simulated_time * server.cost_per_second
            waiting = len(order) - position - 1
            reward_result = self.scheduler.compute_reward(
                actual_execution_time=simulated_time,
                actual_cost=simulated_cost,
                system_load=min(system_load, 1.0),
                tasks_waiting=waiting,
            )
            rollout.append(
                SchedulerExperience(
                    state=bridge_state,
                    action_indices=decision.action_indices,
                    reward=reward_result.reward,
                    done=position == len(order) - 1,
                )
            )
            schedules.append(
                TaskSchedule(
                    task_id=task_id,
                    server_id=server.server_id,
                    predicted_execution_time=prediction.target_exec_time,
                    simulated_execution_time=simulated_time,
                    simulated_cost=simulated_cost,
                    reward=reward_result.reward,
                    execution_time_weight=reward_result.execution_time_weight,
                    cost_weight=reward_result.cost_weight,
                )
            )

        update = self.scheduler.update(rollout) if train_scheduler else None
        return SimulationResult(
            schedules=tuple(schedules),
            embeddings=embeddings,
            scheduler_update=update,
        )


def run_demo(
    *,
    seed: int = 7,
    task_count: int = 8,
    scheduler_episodes: int = 32,
    checkpoint_dir: Optional[str] = None,
) -> SimulationResult:
    """Fit synthetic regressors and run a reproducible, synthetic DAG example."""
    if task_count < 1 or scheduler_episodes < 1:
        raise ValueError("task_count and scheduler_episodes must be positive")
    torch.manual_seed(seed)
    training_tasks, training_contexts = generate_synthetic_training_data(
        sample_count=600, seed=seed
    )
    split = int(len(training_tasks) * 0.8)
    train_tasks, validation_tasks = training_tasks[:split], training_tasks[split:]
    train_contexts = training_contexts[:split]
    validation_contexts = training_contexts[split:]
    predictor = TaskResourcePredictor(random_state=seed)
    predictor.fit_synthetic(train_tasks, train_contexts)

    tasks, contexts = _demo_workload(task_count, seed)
    encoder = TaskGraphEncoder(input_dim=12, hidden_dim=32, embedding_dim=32)
    train_baseline = predictor.predict(train_tasks, train_contexts)
    validation_baseline = predictor.predict(
        validation_tasks, validation_contexts
    )
    train_features = task_gnn_feature_tensor(
        train_tasks, train_contexts, train_baseline
    )
    validation_features = task_gnn_feature_tensor(
        validation_tasks, validation_contexts, validation_baseline
    )
    train_targets = torch.tensor(
        [
            [target[name] for name in (
                "target_exec_time",
                "target_cpu_req",
                "target_gpu_req",
                "target_ram_req",
            )]
            for target in synthetic_resource_targets(
                train_tasks, train_contexts
            )
        ],
        dtype=torch.float32,
    )
    validation_targets = torch.tensor(
        [
            [target[name] for name in (
                "target_exec_time",
                "target_cpu_req",
                "target_gpu_req",
                "target_ram_req",
            )]
            for target in synthetic_resource_targets(
                validation_tasks, validation_contexts
            )
        ],
        dtype=torch.float32,
    )
    train_edges = _synthetic_dag_edges(len(train_tasks), seed)
    validation_edges = _synthetic_dag_edges(len(validation_tasks), seed + 1)
    gnn_metrics = encoder.fit_resource_refiner(
        train_features,
        train_edges,
        train_targets,
        validation_features,
        validation_edges,
        validation_targets,
        epochs=60,
        patience=8,
    )
    telemetry_dim = 7
    scheduler = MARLScheduler(
        input_dim=encoder.embedding_dim + telemetry_dim,
        hidden_dim=128,
        reward_config=AdaptiveRewardConfig(
            execution_time_scale=10.0,
            cost_scale=0.5,
            cost_weight_at_idle=0.7,
            cost_weight_at_max_pressure=0.2,
            queue_pressure_threshold=max(1, task_count // 2),
        ),
    )
    servers = (
        Server("edge-small", 4, 1, 16, 100, 18, 0.015, initial_load=0.2),
        Server("cloud-balanced", 8, 4, 32, 500, 35, 0.035, initial_load=0.1),
        Server("cloud-fast", 16, 8, 64, 900, 48, 0.08, initial_load=0.05),
    )
    engine = OfflineSchedulingEngine(predictor, encoder, scheduler, servers)
    evaluation_tasks, evaluation_contexts = _demo_workload(
        task_count, seed + 3000
    )
    baseline_evaluation = engine.run(
        evaluation_tasks,
        evaluation_contexts,
        train_scheduler=False,
        deterministic=True,
    )
    baseline_evaluation_reward = sum(
        schedule.reward for schedule in baseline_evaluation.schedules
    ) / len(baseline_evaluation.schedules)
    episode_rewards = []
    last_update = {}
    for episode in range(scheduler_episodes):
        episode_tasks, episode_contexts = _demo_workload(
            task_count, seed + 1000 + episode
        )
        episode_result = engine.run(episode_tasks, episode_contexts)
        episode_rewards.append(
            sum(schedule.reward for schedule in episode_result.schedules)
            / len(episode_result.schedules)
        )
        last_update = episode_result.scheduler_update or {}
    trained_evaluation = engine.run(
        evaluation_tasks,
        evaluation_contexts,
        train_scheduler=False,
        deterministic=True,
    )
    trained_evaluation_reward = sum(
        schedule.reward for schedule in trained_evaluation.schedules
    ) / len(trained_evaluation.schedules)
    result = engine.run(
        tasks, contexts, train_scheduler=False, deterministic=True
    )
    scheduler_metrics = {
        **last_update,
        "training_episodes": scheduler_episodes,
        "mean_training_reward": sum(episode_rewards) / len(episode_rewards),
        "last_episode_reward": episode_rewards[-1],
        "baseline_evaluation_reward": baseline_evaluation_reward,
        "trained_evaluation_reward": trained_evaluation_reward,
    }
    result = replace(
        result,
        scheduler_update=scheduler_metrics,
        gnn_metrics=gnn_metrics,
    )

    if checkpoint_dir is not None:
        from pathlib import Path

        output = Path(checkpoint_dir)
        output.mkdir(parents=True, exist_ok=True)
        save_predictor(predictor, output / "task_predictor.joblib")
        save_gnn_encoder(encoder, output / "task_gnn.pt")
        scheduler.save(str(output / "marl_scheduler.pt"))
    return result


def _demo_workload(
    task_count: int, seed: int
) -> Tuple[List[TaskInput], List[InfrastructureContext]]:
    tasks = _demo_tasks(task_count, seed)
    contexts = [
        InfrastructureContext(
            available_bandwidth=200 + 30 * (index % 5),
            latency=12 + 4 * (index % 4),
            available_ram=32,
        )
        for index in range(task_count)
    ]
    return tasks, contexts


def _synthetic_dag_edges(task_count: int, seed: int) -> torch.Tensor:
    """Create separate short DAGs from ordered synthetic workflow samples."""
    from gnn_encoder import build_dependency_edges

    parents = {}
    for index in range(task_count):
        if index % 5 == 0:
            parents[f"sample-{index}"] = ()
        else:
            parents[f"sample-{index}"] = (f"sample-{index - 1}",)
    ids = tuple(parents)
    if seed % 2:
        for index in range(4, task_count, 7):
            parents[f"sample-{index}"] = (f"sample-{index - 2}",)
    return build_dependency_edges(ids, parents)


def _server_telemetry(
    server: Server, remaining: Dict[str, float], max_cost: float
) -> List[float]:
    return [
        remaining["cpu"] / server.cpu_capacity,
        remaining["gpu"] / server.gpu_capacity,
        remaining["ram"] / server.ram_capacity,
        server.bandwidth / 1000.0,
        server.latency / 120.0,
        _server_load(server, remaining),
        server.cost_per_second / max_cost,
    ]


def _server_load(server: Server, remaining: Dict[str, float]) -> float:
    return max(
        server.initial_load,
        1
        - min(
            remaining[resource] / getattr(server, f"{resource}_capacity")
            for resource in ("cpu", "gpu", "ram")
        ),
    )


def _fits(prediction: PredictedTask, remaining: Dict[str, float]) -> bool:
    return (
        prediction.target_cpu_req <= remaining["cpu"]
        and prediction.target_gpu_req <= remaining["gpu"]
        and prediction.target_ram_req <= remaining["ram"]
    )


def _estimate_execution_time(prediction: PredictedTask, server: Server) -> float:
    utilization = max(
        prediction.target_cpu_req / server.cpu_capacity,
        prediction.target_gpu_req / server.gpu_capacity,
        prediction.target_ram_req / server.ram_capacity,
    )
    return prediction.target_exec_time * (
        1 + 0.5 * utilization + server.latency / 1000.0
    )


def _topological_order(
    task_ids: Sequence[str], parents_by_task: Dict[str, Sequence[str]]
) -> List[str]:
    remaining = set(task_ids)
    emitted = set()
    order: List[str] = []
    while remaining:
        ready = [
            task_id
            for task_id in task_ids
            if task_id in remaining
            and all(parent in emitted for parent in parents_by_task.get(task_id, ()))
        ]
        if not ready:
            raise ValueError("task dependencies must form a directed acyclic graph")
        for task_id in ready:
            order.append(task_id)
            emitted.add(task_id)
            remaining.remove(task_id)
    return order


def _demo_tasks(task_count: int, seed: int) -> List[TaskInput]:
    import numpy as np

    rng = np.random.default_rng(seed + 100)
    doc_types = ("text", "image", "video", "archive")
    tasks = []
    for index in range(task_count):
        parents = (f"task-{index - 1}",) if index > 0 and index % 3 != 0 else ()
        tasks.append(
            TaskInput(
                task_id=f"task-{index}",
                file_size=float(rng.uniform(500, 8000)),
                disk_size=float(rng.uniform(1000, 12000)),
                doc_type=doc_types[index % len(doc_types)],
                parents=parents,
                baseline_cpu_req=float(rng.uniform(0, 2.5)),
                baseline_gpu_req=float(rng.uniform(0, 0.8)),
            )
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--checkpoint-dir")
    args = parser.parse_args()
    result = run_demo(
        seed=args.seed, task_count=args.tasks, checkpoint_dir=args.checkpoint_dir
    )
    print("Task allocation simulation")
    for schedule in result.schedules:
        print(
            f"{schedule.task_id:>8} -> {schedule.server_id:<15} "
            f"time={schedule.simulated_execution_time:7.2f}s "
            f"cost={schedule.simulated_cost:7.3f} "
            f"weights(time={schedule.execution_time_weight:.2f}, "
            f"cost={schedule.cost_weight:.2f})"
        )
    print(f"GNN embeddings: {tuple(result.embeddings.shape)}")
    print(f"GNN held-out resource refinement: {result.gnn_metrics}")
    print(f"Scheduler PPO update: {result.scheduler_update}")


if __name__ == "__main__":
    main()
