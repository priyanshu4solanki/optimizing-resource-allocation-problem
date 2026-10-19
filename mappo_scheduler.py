"""MAPPO-style scheduler adapted to the project's task/server bridge state."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from marl_scheduler import (
    AdaptiveReward,
    AdaptiveRewardConfig,
    SchedulerDecision,
    SchedulerExperience,
)
from marl_scheduler_bridge import SchedulerBridgeState


@dataclass(frozen=True)
class MAPPORolloutTransition:
    """One task-agent decision and its centralized training context."""

    observation: torch.Tensor
    global_state: torch.Tensor
    action_mask: torch.Tensor
    action: int
    log_probability: float
    value: float
    reward: float
    done: bool


@dataclass(frozen=True)
class MAPPOAction:
    """Sampled action plus the rollout data required by PPO."""

    action: int
    log_probability: float
    value: float
    observation: torch.Tensor
    global_state: torch.Tensor
    action_mask: torch.Tensor


class _SharedActorCentralCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        global_state_dim: int,
        action_dim: int,
        actor_hidden_dim: int,
        critic_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, 1),
        )

    def forward(
        self, observation: torch.Tensor, global_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.actor(observation), self.critic(global_state).squeeze(-1)


class MAPPOScheduler:
    """Shared task-agent actor with a centralized critic and PPO/GAE updates.

    The policy sees the same ``[GNN embedding, server telemetry]`` bridge
    features as the existing scheduler. Training can additionally enable
    defer/reject actions; the live bridge adapter masks those actions so it
    always returns a feasible worker assignment.
    """

    checkpoint_algorithm = "ctde_mappo_task_scheduler"

    def __init__(
        self,
        input_dim: int,
        num_servers: int,
        observation_dim: int,
        global_state_dim: int,
        *,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.001,
        update_epochs: int = 4,
        minibatches: int = 4,
        reward_config: Optional[AdaptiveRewardConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if min(input_dim, num_servers, observation_dim, global_state_dim) <= 0:
            raise ValueError("scheduler dimensions must be positive")
        if observation_dim != num_servers * input_dim:
            raise ValueError(
                "observation_dim must equal num_servers * bridge input_dim"
            )
        if global_state_dim < observation_dim + 12:
            raise ValueError(
                "global_state_dim must fit one observation, four status values, "
                "and eight global metrics"
            )
        if actor_hidden_dim <= 0 or critic_hidden_dim <= 0:
            raise ValueError("hidden dimensions must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= gamma <= 1 or not 0 <= gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be between 0 and 1")
        if not 0 < clip_epsilon < 1:
            raise ValueError("clip_epsilon must be between 0 and 1")
        if update_epochs <= 0 or minibatches <= 0:
            raise ValueError("update_epochs and minibatches must be positive")

        self.input_dim = input_dim
        self.num_servers = num_servers
        self.observation_dim = observation_dim
        self.global_state_dim = global_state_dim
        self.action_dim = num_servers + 2
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.update_epochs = update_epochs
        self.minibatches = minibatches
        self.reward_config = reward_config or AdaptiveRewardConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = _SharedActorCentralCritic(
            observation_dim,
            global_state_dim,
            self.action_dim,
            actor_hidden_dim,
            critic_hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _bridge_observations(
        self, state: SchedulerBridgeState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint = state.joint_state
        mask = state.feasibility_mask
        if joint.ndim != 3 or joint.shape[1:] != (self.num_servers, self.input_dim):
            raise ValueError(
                "joint_state must have shape "
                f"[num_tasks, {self.num_servers}, {self.input_dim}]"
            )
        if mask.shape != joint.shape[:2]:
            raise ValueError("feasibility_mask must match joint_state task/server axes")
        if not torch.isfinite(joint).all():
            raise ValueError("joint_state must contain only finite values")
        if not mask.any(dim=1).all():
            raise ValueError("every task must have at least one feasible server")
        return (
            joint.to(device=self.device, dtype=torch.float32).flatten(start_dim=1),
            mask.to(device=self.device, dtype=torch.bool),
        )

    def _global_from_observation(self, observation: torch.Tensor) -> torch.Tensor:
        """Build the inference critic context from normalized telemetry summaries."""
        telemetry = observation.reshape(-1, self.num_servers, self.input_dim)[
            :, :, self.input_dim - 7 :
        ]
        loads = telemetry[:, :, 5]
        summary = torch.stack(
            (
                loads.mean(dim=1),
                loads.std(dim=1, unbiased=False),
                loads.max(dim=1).values,
                loads.min(dim=1).values,
                loads.new_zeros(loads.shape[0]),
                loads.new_zeros(loads.shape[0]),
                loads.new_ones(loads.shape[0]),
                loads.new_zeros(loads.shape[0]),
            ),
            dim=1,
        )
        global_state = observation.new_zeros((observation.shape[0], self.global_state_dim))
        global_state[:, : self.observation_dim] = observation
        global_state[:, self.observation_dim] = 1.0
        global_state[:, -8:] = summary
        return global_state

    @torch.no_grad()
    def act(
        self,
        observation: torch.Tensor,
        global_state: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> MAPPOAction:
        """Choose a masked action and capture its old-policy rollout data."""
        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).reshape(-1)
        global_state = torch.as_tensor(
            global_state, dtype=torch.float32, device=self.device
        ).reshape(-1)
        action_mask = torch.as_tensor(
            action_mask, dtype=torch.bool, device=self.device
        ).reshape(-1)
        if observation.numel() != self.observation_dim:
            raise ValueError("observation has the wrong feature dimension")
        if global_state.numel() != self.global_state_dim:
            raise ValueError("global_state has the wrong feature dimension")
        if action_mask.numel() != self.action_dim:
            raise ValueError("action_mask has the wrong action dimension")
        if not action_mask.any():
            raise ValueError("at least one action must be valid")
        if not torch.isfinite(observation).all() or not torch.isfinite(global_state).all():
            raise ValueError("observations must contain only finite values")

        logits, value = self.model(observation.unsqueeze(0), global_state.unsqueeze(0))
        logits = logits.masked_fill(~action_mask.unsqueeze(0), torch.finfo(logits.dtype).min)
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return MAPPOAction(
            action=int(action.item()),
            log_probability=float(distribution.log_prob(action).item()),
            value=float(value.item()),
            observation=observation.detach().clone(),
            global_state=global_state.detach().clone(),
            action_mask=action_mask.detach().clone(),
        )

    @torch.no_grad()
    def select_actions(
        self, state: SchedulerBridgeState, *, deterministic: bool = False
    ) -> SchedulerDecision:
        """Choose feasible server assignments through the existing bridge API."""
        observations, server_mask = self._bridge_observations(state)
        action_mask = torch.zeros(
            (observations.shape[0], self.action_dim),
            dtype=torch.bool,
            device=self.device,
        )
        action_mask[:, : self.num_servers] = server_mask
        global_states = self._global_from_observation(observations)
        logits, values = self.model(observations, global_states)
        logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        distributions = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distributions.sample()
        indices = tuple(int(index) for index in actions.tolist())
        assignments = tuple(
            (task_id, state.server_ids[server_index])
            for task_id, server_index in zip(state.task_ids, indices)
        )
        log_probabilities = tuple(
            float(value) for value in distributions.log_prob(actions).tolist()
        )
        return SchedulerDecision(
            action_indices=indices,
            assignments=assignments,
            log_probabilities=log_probabilities,
            value=float(values.mean().item()),
        )

    def compute_reward(
        self,
        *,
        actual_execution_time: float,
        actual_cost: float,
        system_load: float,
        tasks_waiting: int,
    ) -> AdaptiveReward:
        """Keep the existing live-feedback reward contract intact."""
        if not isfinite(actual_execution_time) or actual_execution_time < 0:
            raise ValueError("actual_execution_time must be finite and non-negative")
        if not isfinite(actual_cost) or actual_cost < 0:
            raise ValueError("actual_cost must be finite and non-negative")
        if not isfinite(system_load) or not 0 <= system_load <= 1:
            raise ValueError("system_load must be finite and between 0 and 1")
        if isinstance(tasks_waiting, bool) or not isinstance(tasks_waiting, int) or tasks_waiting < 0:
            raise ValueError("tasks_waiting must be a non-negative integer")
        config = self.reward_config
        pressure = max(
            system_load,
            min(tasks_waiting / config.queue_pressure_threshold, 1.0),
        )
        cost_weight = config.cost_weight_at_idle + pressure * (
            config.cost_weight_at_max_pressure - config.cost_weight_at_idle
        )
        time_weight = 1 - cost_weight
        reward = 1 - (
            time_weight * actual_execution_time / config.execution_time_scale
            + cost_weight * actual_cost / config.cost_scale
        )
        return AdaptiveReward(reward, time_weight, cost_weight, pressure)

    def update(self, experiences: Sequence[object]) -> dict:
        """Apply clipped PPO with GAE to an episode or a live feedback batch."""
        if not experiences:
            raise ValueError("at least one scheduler experience is required")

        transitions = []
        for experience in experiences:
            if isinstance(experience, MAPPORolloutTransition):
                transitions.append(experience)
                continue
            if not isinstance(experience, SchedulerExperience):
                raise TypeError("unsupported scheduler experience type")
            observations, masks = self._bridge_observations(experience.state)
            global_states = self._global_from_observation(observations)
            with torch.no_grad():
                logits, values = self.model(observations, global_states)
                valid_actions = torch.zeros_like(logits, dtype=torch.bool)
                valid_actions[:, : self.num_servers] = masks
                logits = logits.masked_fill(
                    ~valid_actions, torch.finfo(logits.dtype).min
                )
                distribution = Categorical(logits=logits)
                actions = torch.tensor(
                    experience.action_indices, dtype=torch.long, device=self.device
                )
                if actions.shape != (observations.shape[0],):
                    raise ValueError("experience action count must match task count")
                if (actions < 0).any() or (actions >= self.num_servers).any():
                    raise ValueError("experience action is outside the server range")
                if not valid_actions.gather(1, actions.unsqueeze(1)).all():
                    raise ValueError("experience contains an infeasible server action")
                log_probs = distribution.log_prob(actions)
            for index in range(observations.shape[0]):
                transitions.append(
                    MAPPORolloutTransition(
                        observation=observations[index].detach().clone(),
                        global_state=global_states[index].detach().clone(),
                        action_mask=valid_actions[index].detach().clone(),
                        action=int(actions[index].item()),
                        log_probability=float(log_probs[index].item()),
                        value=float(values[index].item()),
                        reward=float(experience.reward),
                        done=bool(experience.done),
                    )
                )

        for transition in transitions:
            if not isfinite(transition.reward):
                raise ValueError("experience reward must be finite")
            if transition.action_mask.shape != (self.action_dim,):
                raise ValueError("experience action mask has the wrong size")
            if not transition.action_mask[transition.action]:
                raise ValueError("experience action is masked as invalid")
        if not transitions[-1].done:
            raise ValueError("the rollout must end at an episode boundary")

        observations = torch.stack(
            [item.observation.to(self.device) for item in transitions]
        )
        global_states = torch.stack(
            [item.global_state.to(self.device) for item in transitions]
        )
        masks = torch.stack([item.action_mask.to(self.device) for item in transitions])
        actions = torch.tensor(
            [item.action for item in transitions], dtype=torch.long, device=self.device
        )
        old_log_probs = torch.tensor(
            [item.log_probability for item in transitions],
            dtype=torch.float32,
            device=self.device,
        )
        values = torch.tensor(
            [item.value for item in transitions], dtype=torch.float32, device=self.device
        )
        rewards = torch.tensor(
            [item.reward for item in transitions], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [item.done for item in transitions], dtype=torch.float32, device=self.device
        )
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros((), dtype=torch.float32, device=self.device)
        next_value = torch.zeros((), dtype=torch.float32, device=self.device)
        for index in range(len(transitions) - 1, -1, -1):
            continuation = 1.0 - dones[index]
            delta = rewards[index] + self.gamma * next_value * continuation - values[index]
            gae = delta + self.gamma * self.gae_lambda * continuation * gae
            advantages[index] = gae
            next_value = values[index]
        returns = advantages + values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        entropy_value = 0.0
        batch_size = len(transitions)
        for _ in range(self.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for indices in torch.tensor_split(
                permutation, min(self.minibatches, batch_size)
            ):
                logits, predicted_values = self.model(
                    observations[indices], global_states[indices]
                )
                logits = logits.masked_fill(
                    ~masks[indices], torch.finfo(logits.dtype).min
                )
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(actions[indices])
                ratios = (new_log_probs - old_log_probs[indices]).exp()
                unclipped = ratios * advantages[indices]
                clipped = ratios.clamp(
                    1 - self.clip_epsilon, 1 + self.clip_epsilon
                ) * advantages[indices]
                actor_loss = -torch.minimum(unclipped, clipped).mean()
                critic_loss = nn.functional.mse_loss(
                    predicted_values, returns[indices]
                )
                entropy = distribution.entropy().mean()
                loss = (
                    actor_loss
                    + self.value_coefficient * critic_loss
                    - self.entropy_coefficient * entropy
                )
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                actor_loss_value = float(actor_loss.detach().item())
                critic_loss_value = float(critic_loss.detach().item())
                entropy_value = float(entropy.detach().item())

        return {
            "actor_loss": actor_loss_value,
            "critic_loss": critic_loss_value,
            "entropy": entropy_value,
            "transitions": batch_size,
        }

    def save(self, path: str) -> None:
        """Persist MAPPO configuration, weights, and optimizer state."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.checkpoint_algorithm,
                "input_dim": self.input_dim,
                "num_servers": self.num_servers,
                "observation_dim": self.observation_dim,
                "global_state_dim": self.global_state_dim,
                "actor_hidden_dim": self.actor_hidden_dim,
                "critic_hidden_dim": self.critic_hidden_dim,
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_epsilon": self.clip_epsilon,
                "value_coefficient": self.value_coefficient,
                "entropy_coefficient": self.entropy_coefficient,
                "update_epochs": self.update_epochs,
                "minibatches": self.minibatches,
                "reward_config": {
                    "execution_time_scale": self.reward_config.execution_time_scale,
                    "cost_scale": self.reward_config.cost_scale,
                    "cost_weight_at_idle": self.reward_config.cost_weight_at_idle,
                    "cost_weight_at_max_pressure": (
                        self.reward_config.cost_weight_at_max_pressure
                    ),
                    "queue_pressure_threshold": (
                        self.reward_config.queue_pressure_threshold
                    ),
                },
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            target,
        )

    @classmethod
    def load(
        cls, path: str, *, device: Optional[torch.device] = None
    ) -> "MAPPOScheduler":
        """Restore an adapted MAPPO checkpoint."""
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        checkpoint = torch.load(
            Path(path), map_location=target_device, weights_only=True
        )
        if checkpoint.get("algorithm") != cls.checkpoint_algorithm:
            raise ValueError("checkpoint is not an adapted MAPPO scheduler")
        scheduler = cls(
            input_dim=checkpoint["input_dim"],
            num_servers=checkpoint["num_servers"],
            observation_dim=checkpoint["observation_dim"],
            global_state_dim=checkpoint["global_state_dim"],
            actor_hidden_dim=checkpoint["actor_hidden_dim"],
            critic_hidden_dim=checkpoint["critic_hidden_dim"],
            learning_rate=checkpoint["learning_rate"],
            gamma=checkpoint["gamma"],
            gae_lambda=checkpoint["gae_lambda"],
            clip_epsilon=checkpoint["clip_epsilon"],
            value_coefficient=checkpoint["value_coefficient"],
            entropy_coefficient=checkpoint["entropy_coefficient"],
            update_epochs=checkpoint["update_epochs"],
            minibatches=checkpoint["minibatches"],
            reward_config=AdaptiveRewardConfig(**checkpoint["reward_config"]),
            device=target_device,
        )
        scheduler.model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return scheduler


def load_scheduler_checkpoint(
    path: str, *, device: Optional[torch.device] = None
) -> object:
    """Load either the original PPO scheduler or a MAPPO scheduler checkpoint."""
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    checkpoint = torch.load(Path(path), map_location=target_device, weights_only=True)
    if checkpoint.get("algorithm") == MAPPOScheduler.checkpoint_algorithm:
        return MAPPOScheduler.load(path, device=target_device)
    from marl_scheduler import MARLScheduler

    return MARLScheduler.load(path, device=target_device)


__all__ = [
    "MAPPOAction",
    "MAPPORolloutTransition",
    "MAPPOScheduler",
    "load_scheduler_checkpoint",
]
