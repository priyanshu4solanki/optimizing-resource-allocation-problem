"""Build task/server candidate states for a MARL scheduler."""

from dataclasses import dataclass
from typing import Hashable, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SchedulerBridgeState:
    """Joint candidate states and the action-index metadata they require."""

    joint_state: torch.Tensor
    feasibility_mask: torch.Tensor
    task_ids: Tuple[Hashable, ...]
    server_ids: Tuple[Hashable, ...]


def build_joint_state(
    task_embeddings: torch.Tensor,
    server_telemetry: torch.Tensor,
    *,
    task_ids: Optional[Sequence[Hashable]] = None,
    server_ids: Optional[Sequence[Hashable]] = None,
    feasibility_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> SchedulerBridgeState:
    """Combine every task embedding with every server's telemetry.

    ``joint_state[i, j]`` is the candidate feature vector for assigning task
    ``i`` to server ``j``. Its features are ordered as
    ``[task_embedding, server_telemetry]``. The returned mask uses the same
    task/server ordering; ``False`` marks an unavailable or otherwise invalid
    assignment, but does not remove that candidate from the tensor.
    """
    task_embeddings = torch.as_tensor(task_embeddings)
    server_telemetry = torch.as_tensor(server_telemetry)

    if task_embeddings.ndim != 2:
        raise ValueError("task_embeddings must have shape [num_tasks, embedding_dim]")
    if server_telemetry.ndim != 2:
        raise ValueError("server_telemetry must have shape [num_servers, telemetry_dim]")
    if not task_embeddings.is_floating_point():
        raise TypeError("task_embeddings must contain floating-point values")
    if not server_telemetry.is_floating_point():
        raise TypeError("server_telemetry must contain floating-point values")
    if task_embeddings.shape[0] == 0:
        raise ValueError("at least one task embedding is required")
    if server_telemetry.shape[0] == 0:
        raise ValueError("at least one server telemetry row is required")
    if not torch.isfinite(task_embeddings).all():
        raise ValueError("task_embeddings must contain only finite values")
    if not torch.isfinite(server_telemetry).all():
        raise ValueError("server_telemetry must contain only finite values")

    target_device = torch.device(device) if device is not None else task_embeddings.device
    target_dtype = torch.promote_types(task_embeddings.dtype, server_telemetry.dtype)
    task_embeddings = task_embeddings.to(device=target_device, dtype=target_dtype)
    server_telemetry = server_telemetry.to(device=target_device, dtype=target_dtype)

    num_tasks, embedding_dim = task_embeddings.shape
    num_servers, telemetry_dim = server_telemetry.shape

    if task_ids is None:
        task_ids = tuple(range(num_tasks))
    else:
        task_ids = tuple(task_ids)
    if server_ids is None:
        server_ids = tuple(range(num_servers))
    else:
        server_ids = tuple(server_ids)
    if len(task_ids) != num_tasks:
        raise ValueError("task_ids length must match the number of task embeddings")
    if len(server_ids) != num_servers:
        raise ValueError("server_ids length must match the number of telemetry rows")

    task_features = task_embeddings.unsqueeze(1).expand(
        num_tasks, num_servers, embedding_dim
    )
    server_features = server_telemetry.unsqueeze(0).expand(
        num_tasks, num_servers, telemetry_dim
    )
    joint_state = torch.cat((task_features, server_features), dim=-1)

    if feasibility_mask is None:
        feasibility_mask = torch.ones(
            (num_tasks, num_servers), dtype=torch.bool, device=target_device
        )
    else:
        feasibility_mask = torch.as_tensor(
            feasibility_mask, dtype=torch.bool, device=target_device
        )
        if feasibility_mask.shape != (num_tasks, num_servers):
            raise ValueError(
                "feasibility_mask must have shape [num_tasks, num_servers]"
            )

    return SchedulerBridgeState(
        joint_state=joint_state,
        feasibility_mask=feasibility_mask,
        task_ids=task_ids,
        server_ids=server_ids,
    )
