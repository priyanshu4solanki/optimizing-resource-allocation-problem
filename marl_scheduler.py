"""Shared-policy MARL scheduler for task-to-server allocation."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Categorical

from marl_scheduler_bridge import SchedulerBridgeState


@dataclass(frozen=True)
class SchedulerDecision:
    """Task assignments selected from a bridge state."""

    action_indices: Tuple[int, ...]
    assignments: Tuple[Tuple[object, object], ...]
    log_probabilities: Tuple[float, ...]
    value: float


@dataclass(frozen=True)
class SchedulerExperience:
    """One joint scheduler transition for PPO training."""

    state: SchedulerBridgeState
    action_indices: Tuple[int, ...]
    reward: float
    done: bool


@dataclass(frozen=True)
class AdaptiveRewardConfig:
    """Reward scales and cost-weight bounds for changing workload pressure."""

    execution_time_scale: float = 1.0
    cost_scale: float = 1.0
    cost_weight_at_idle: float = 0.7
    cost_weight_at_max_pressure: float = 0.2
    queue_pressure_threshold: int = 20

    def __post_init__(self) -> None:
        if not isfinite(self.execution_time_scale) or self.execution_time_scale <= 0:
            raise ValueError("execution_time_scale must be finite and positive")
        if not isfinite(self.cost_scale) or self.cost_scale <= 0:
            raise ValueError("cost_scale must be finite and positive")
        if not 0 <= self.cost_weight_at_idle <= 1:
            raise ValueError("cost_weight_at_idle must be between 0 and 1")
        if not 0 <= self.cost_weight_at_max_pressure <= 1:
            raise ValueError("cost_weight_at_max_pressure must be between 0 and 1")
        if self.queue_pressure_threshold <= 0:
            raise ValueError("queue_pressure_threshold must be positive")


@dataclass(frozen=True)
class AdaptiveReward:
    """Reward and diagnostic weights computed for a completed task."""

    reward: float
    execution_time_weight: float
    cost_weight: float
    workload_pressure: float


class _ActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, joint_state: torch.Tensor, feasibility_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Each task uses the same actor; the critic observes the joint candidate set.
        logits = self.actor(joint_state).squeeze(-1)
        weights = feasibility_mask.to(dtype=joint_state.dtype).unsqueeze(-1)
        pooled_state = (joint_state * weights).sum(dim=(1, 2))
        pooled_state = pooled_state / weights.sum(dim=(1, 2)).clamp_min(1.0)
        value = self.critic(pooled_state).squeeze(-1)
        return logits, value


class MARLScheduler:
    """Decentralized shared actor with a centralized critic, trained using PPO.

    One policy is shared by all task agents. Each task independently selects a
    feasible server from its pairwise candidate states. Feasibility constraints
    such as resource capacity should be reflected in the bridge mask.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coefficient: float = 0.5,
        entropy_coefficient: float = 0.01,
        update_epochs: int = 4,
        reward_config: Optional[AdaptiveRewardConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not 0 < clip_epsilon < 1:
            raise ValueError("clip_epsilon must be between 0 and 1")
        if update_epochs <= 0:
            raise ValueError("update_epochs must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.update_epochs = update_epochs
        self.reward_config = reward_config or AdaptiveRewardConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = _ActorCritic(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

    def compute_reward(
        self,
        *,
        actual_execution_time: float,
        actual_cost: float,
        system_load: float,
        tasks_waiting: int,
    ) -> AdaptiveReward:
        """Score an outcome, prioritizing speed more as load or queue pressure rises.

        ``system_load`` must be normalized to [0, 1]. The pressure is the
        larger of normalized system load and queue length relative to the
        configured threshold. Measure both before dispatching the task.
        """
        if not isfinite(actual_execution_time) or actual_execution_time < 0:
            raise ValueError("actual_execution_time must be finite and non-negative")
        if not isfinite(actual_cost) or actual_cost < 0:
            raise ValueError("actual_cost must be finite and non-negative")
        if not isfinite(system_load) or not 0 <= system_load <= 1:
            raise ValueError("system_load must be finite and between 0 and 1")
        if (
            isinstance(tasks_waiting, bool)
            or not isinstance(tasks_waiting, Integral)
            or tasks_waiting < 0
        ):
            raise ValueError("tasks_waiting must be a non-negative integer")

        pressure_from_queue = min(
            tasks_waiting / self.reward_config.queue_pressure_threshold, 1.0
        )
        pressure = max(system_load, pressure_from_queue)
        config = self.reward_config
        cost_weight = config.cost_weight_at_idle + pressure * (
            config.cost_weight_at_max_pressure - config.cost_weight_at_idle
        )
        execution_time_weight = 1.0 - cost_weight
        normalized_time = actual_execution_time / config.execution_time_scale
        normalized_cost = actual_cost / config.cost_scale
        reward = 1.0 - (
            execution_time_weight * normalized_time + cost_weight * normalized_cost
        )
        return AdaptiveReward(
            reward=reward,
            execution_time_weight=execution_time_weight,
            cost_weight=cost_weight,
            workload_pressure=pressure,
        )

    def _prepare_state(
        self, state: SchedulerBridgeState
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        joint_state = state.joint_state
        mask = state.feasibility_mask
        if joint_state.ndim != 3:
            raise ValueError("joint_state must have shape [num_tasks, num_servers, features]")
        if joint_state.shape[-1] != self.input_dim:
            raise ValueError(
                f"joint_state feature size {joint_state.shape[-1]} "
                f"does not match scheduler input_dim {self.input_dim}"
            )
        if mask.shape != joint_state.shape[:2]:
            raise ValueError("feasibility_mask must match the task/server dimensions")
        if not mask.any(dim=1).all():
            raise ValueError("every task must have at least one feasible server")
        if not torch.isfinite(joint_state).all():
            raise ValueError("joint_state must contain only finite values")
        return (
            joint_state.to(device=self.device, dtype=torch.float32).unsqueeze(0),
            mask.to(device=self.device, dtype=torch.bool).unsqueeze(0),
        )

    @torch.no_grad()
    def select_actions(
        self, state: SchedulerBridgeState, *, deterministic: bool = False
    ) -> SchedulerDecision:
        """Select one server per task and return assignments in bridge ID order."""
        joint_state, mask = self._prepare_state(state)
        logits, value = self.model(joint_state, mask)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        log_probabilities = distribution.log_prob(actions).squeeze(0)
        action_indices = tuple(int(index) for index in actions.squeeze(0).tolist())
        assignments = tuple(
            (task_id, state.server_ids[server_index])
            for task_id, server_index in zip(state.task_ids, action_indices)
        )
        return SchedulerDecision(
            action_indices=action_indices,
            assignments=assignments,
            log_probabilities=tuple(
                float(log_prob) for log_prob in log_probabilities.tolist()
            ),
            value=float(value.item()),
        )

    def update(self, experiences: Sequence[SchedulerExperience]) -> dict:
        """Run a PPO update over chronological transitions from complete episodes."""
        if not experiences:
            raise ValueError("at least one scheduler experience is required")
        if not experiences[-1].done:
            raise ValueError("the rollout must end at an episode boundary")

        states: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        actions: List[torch.Tensor] = []
        returns: List[float] = [0.0] * len(experiences)

        for index, experience in enumerate(experiences):
            joint_state, feasibility_mask = self._prepare_state(experience.state)
            if len(experience.action_indices) != joint_state.shape[1]:
                raise ValueError("each experience needs one action per task")
            action = torch.tensor(
                experience.action_indices, dtype=torch.long, device=self.device
            )
            if (action < 0).any() or (action >= joint_state.shape[2]).any():
                raise ValueError("action index is outside the server range")
            if not feasibility_mask[0].gather(1, action.unsqueeze(1)).all():
                raise ValueError("experience contains an infeasible task/server action")
            if not torch.isfinite(torch.tensor(experience.reward)):
                raise ValueError("experience reward must be finite")
            states.append(joint_state.squeeze(0))
            masks.append(feasibility_mask.squeeze(0))
            actions.append(action)

        discounted_return = 0.0
        for index in range(len(experiences) - 1, -1, -1):
            experience = experiences[index]
            if experience.done:
                discounted_return = 0.0
            discounted_return = experience.reward + self.gamma * discounted_return
            returns[index] = discounted_return

        joint_states = torch.stack(states)
        feasibility_masks = torch.stack(masks)
        action_batch = torch.stack(actions)
        return_batch = torch.tensor(returns, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            old_logits, old_values = self.model(joint_states, feasibility_masks)
            old_logits = old_logits.masked_fill(
                ~feasibility_masks, torch.finfo(old_logits.dtype).min
            )
            old_distributions = Categorical(logits=old_logits)
            old_log_probs = old_distributions.log_prob(action_batch).sum(dim=1)
            advantages = return_batch - old_values
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )

        actor_loss_value = 0.0
        critic_loss_value = 0.0
        entropy_value = 0.0
        for _ in range(self.update_epochs):
            logits, values = self.model(joint_states, feasibility_masks)
            logits = logits.masked_fill(
                ~feasibility_masks, torch.finfo(logits.dtype).min
            )
            distributions = Categorical(logits=logits)
            new_log_probs = distributions.log_prob(action_batch).sum(dim=1)
            ratios = (new_log_probs - old_log_probs).exp()
            unclipped = ratios * advantages
            clipped = ratios.clamp(
                1 - self.clip_epsilon, 1 + self.clip_epsilon
            ) * advantages
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            critic_loss = nn.functional.mse_loss(values, return_batch)
            entropy = distributions.entropy().sum(dim=1).mean()
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
            "transitions": len(experiences),
        }

    def save(self, path: str) -> None:
        """Persist scheduler weights, optimizer state, and configuration."""
        checkpoint = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "clip_epsilon": self.clip_epsilon,
            "value_coefficient": self.value_coefficient,
            "entropy_coefficient": self.entropy_coefficient,
            "update_epochs": self.update_epochs,
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
        }
        torch.save(checkpoint, Path(path))

    @classmethod
    def load(
        cls, path: str, *, device: Optional[torch.device] = None
    ) -> "MARLScheduler":
        """Restore a scheduler checkpoint produced by :meth:`save`."""
        target_device = torch.device(device) if device is not None else torch.device("cpu")
        checkpoint = torch.load(
            Path(path), map_location=target_device, weights_only=True
        )
        scheduler = cls(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            learning_rate=checkpoint["learning_rate"],
            gamma=checkpoint["gamma"],
            clip_epsilon=checkpoint["clip_epsilon"],
            value_coefficient=checkpoint["value_coefficient"],
            entropy_coefficient=checkpoint["entropy_coefficient"],
            update_epochs=checkpoint["update_epochs"],
            reward_config=AdaptiveRewardConfig(**checkpoint.get("reward_config", {})),
            device=target_device,
        )
        scheduler.model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return scheduler
