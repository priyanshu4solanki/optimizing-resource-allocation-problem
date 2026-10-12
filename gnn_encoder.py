"""Graph encoder for task resource features and DAG dependencies."""

from copy import deepcopy
from typing import Dict, Sequence, Tuple

import torch
from torch import nn

try:
    from torch_geometric.nn import GCNConv
except ImportError:
    GCNConv = None

RESOURCE_DIM = 4


class _FallbackGCNLayer(nn.Module):
    """Directed mean-neighbor graph convolution used when PyG is unavailable."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        node_count = features.shape[0]
        source, target = edge_index
        aggregated = features.new_zeros(features.shape)
        if source.numel():
            aggregated.index_add_(0, target, features[source])
        degree = features.new_ones((node_count, 1))
        if target.numel():
            degree.index_add_(
                0, target, features.new_ones((target.shape[0], 1))
            )
        aggregated = (aggregated + features) / degree
        return self.linear(aggregated)


class TaskGraphEncoder(nn.Module):
    """Encode predicted task resources in the context of their dependency DAG."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 32,
        embedding_dim: int = 32,
        *,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or embedding_dim <= 0:
            raise ValueError("encoder dimensions must be positive")
        if backend not in ("auto", "pyg", "fallback"):
            raise ValueError("backend must be 'auto', 'pyg', or 'fallback'")
        if backend == "pyg" and GCNConv is None:
            raise ImportError("backend='pyg' requires torch_geometric")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = RESOURCE_DIM
        self.backend = "pyg" if backend == "pyg" or (backend == "auto" and GCNConv) else "fallback"

        if self.backend == "pyg":
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, embedding_dim)
        else:
            self.conv1 = _FallbackGCNLayer(input_dim, hidden_dim)
            self.conv2 = _FallbackGCNLayer(hidden_dim, embedding_dim)
        self.activation = nn.ReLU()
        self.resource_head = nn.Linear(embedding_dim, self.output_dim)
        nn.init.zeros_(self.resource_head.weight)
        nn.init.zeros_(self.resource_head.bias)
        self.register_buffer("resource_residual_scale", torch.ones(RESOURCE_DIM))

    def forward(
        self, task_features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        if task_features.ndim != 2 or task_features.shape[1] != self.input_dim:
            raise ValueError(
                f"task_features must have shape [num_tasks, {self.input_dim}]"
            )
        if task_features.shape[0] == 0:
            raise ValueError("at least one task feature row is required")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if edge_index.dtype not in (torch.int32, torch.int64):
            raise TypeError("edge_index must contain integer node indices")
        if not torch.isfinite(task_features).all():
            raise ValueError("task_features must contain only finite values")
        edge_index = edge_index.to(device=task_features.device, dtype=torch.long)
        if edge_index.numel() and (
            (edge_index < 0).any() or (edge_index >= task_features.shape[0]).any()
        ):
            raise ValueError("edge_index contains an out-of-range task index")
        hidden = self.activation(self.conv1(task_features, edge_index))
        return self.conv2(hidden, edge_index)

    def encode(
        self, task_features: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Create embeddings for scheduler inference without changing train mode."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                return self.forward(task_features, edge_index)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def predict_resources(
        self, task_features: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Refine baseline resource estimates and return their GNN embeddings."""
        embeddings = self.encode(task_features, edge_index)
        baseline = task_features[:, :RESOURCE_DIM].to(
            device=embeddings.device, dtype=embeddings.dtype
        )
        residual = self.resource_head(embeddings) * self.resource_residual_scale
        predictions = torch.clamp(baseline + residual, min=0.0)
        return predictions, embeddings

    def fit_resource_refiner(
        self,
        train_features: torch.Tensor,
        train_edge_index: torch.Tensor,
        train_targets: torch.Tensor,
        validation_features: torch.Tensor,
        validation_edge_index: torch.Tensor,
        validation_targets: torch.Tensor,
        *,
        epochs: int = 200,
        learning_rate: float = 1e-3,
        patience: int = 20,
    ) -> Dict[str, float]:
        """Supervise the graph residual head on labeled four-resource examples."""
        if epochs <= 0 or learning_rate <= 0 or patience <= 0:
            raise ValueError("epochs, learning_rate, and patience must be positive")
        train_features, train_edge_index, train_targets = _validate_resource_data(
            train_features, train_edge_index, train_targets, self.input_dim
        )
        validation_features, validation_edge_index, validation_targets = (
            _validate_resource_data(
                validation_features,
                validation_edge_index,
                validation_targets,
                self.input_dim,
            )
        )
        if train_features.device != validation_features.device:
            raise ValueError("training and validation tensors must be on the same device")

        device = next(self.parameters()).device
        train_features = train_features.to(device=device, dtype=torch.float32)
        train_edge_index = train_edge_index.to(device=device, dtype=torch.long)
        train_targets = train_targets.to(device=device, dtype=torch.float32)
        validation_features = validation_features.to(device=device, dtype=torch.float32)
        validation_edge_index = validation_edge_index.to(
            device=device, dtype=torch.long
        )
        validation_targets = validation_targets.to(
            device=device, dtype=torch.float32
        )

        baseline_train = train_features[:, :RESOURCE_DIM]
        baseline_validation = validation_features[:, :RESOURCE_DIM]
        train_residual = train_targets - baseline_train
        residual_scale = train_residual.std(dim=0, unbiased=False).clamp_min(1e-3)
        self.resource_residual_scale.copy_(residual_scale)
        train_target = train_residual / residual_scale
        validation_target = (
            validation_targets - baseline_validation
        ) / residual_scale
        baseline_loss = validation_target.square().mean()
        baseline_mae = (
            baseline_validation - validation_targets
        ).abs().mean()

        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        best_loss = float(baseline_loss.item())
        best_state = deepcopy(self.state_dict())
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            self.train()
            optimizer.zero_grad()
            train_embeddings = self.forward(train_features, train_edge_index)
            train_output = self.resource_head(train_embeddings)
            train_loss = nn.functional.smooth_l1_loss(train_output, train_target)
            train_loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optimizer.step()

            self.eval()
            with torch.no_grad():
                validation_embeddings = self.forward(
                    validation_features, validation_edge_index
                )
                validation_output = self.resource_head(validation_embeddings)
                validation_loss = nn.functional.mse_loss(
                    validation_output, validation_target
                )
            candidate_loss = float(validation_loss.item())
            if candidate_loss < best_loss - 1e-8:
                best_loss = candidate_loss
                best_state = deepcopy(self.state_dict())
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break

        self.load_state_dict(best_state)
        self.eval()
        with torch.no_grad():
            predictions, _ = self.predict_resources(
                validation_features, validation_edge_index
            )
            refined_mae = (predictions - validation_targets).abs().mean()
        refiner_enabled = bool(refined_mae <= baseline_mae + 1e-7)
        if not refiner_enabled:
            nn.init.zeros_(self.resource_head.weight)
            nn.init.zeros_(self.resource_head.bias)
        return {
            "baseline_normalized_mse": float(baseline_loss.item()),
            "refined_normalized_mse": best_loss,
            "baseline_mae": float(baseline_mae.item()),
            "refined_mae": float(
                refined_mae.item() if refiner_enabled else baseline_mae.item()
            ),
            "best_epoch": float(best_epoch),
            "resource_refiner_enabled": float(refiner_enabled),
        }


def build_dependency_edges(
    task_ids: Sequence[str], parents_by_task: Dict[str, Sequence[str]]
) -> torch.Tensor:
    """Validate task dependencies and return parent-to-child ``edge_index``."""
    ids = tuple(task_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("task_ids must be unique")
    index_by_id = {task_id: index for index, task_id in enumerate(ids)}
    edges = []
    indegree = [0] * len(ids)
    children = [[] for _ in ids]
    for child_id in ids:
        for parent_id in parents_by_task.get(child_id, ()):
            if parent_id not in index_by_id:
                raise ValueError(
                    f"task {child_id!r} depends on unknown task {parent_id!r}"
                )
            source, target = index_by_id[parent_id], index_by_id[child_id]
            if source == target:
                raise ValueError(f"task {child_id!r} cannot depend on itself")
            edges.append((source, target))
            children[source].append(target)
            indegree[target] += 1

    ready = [index for index, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(ids):
        raise ValueError("task dependencies must form a directed acyclic graph")

    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _validate_resource_data(
    features: torch.Tensor,
    edge_index: torch.Tensor,
    targets: torch.Tensor,
    input_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or features.shape[1] != input_dim or features.shape[0] == 0:
        raise ValueError(f"features must have shape [num_tasks, {input_dim}]")
    if targets.ndim != 2 or targets.shape != (features.shape[0], RESOURCE_DIM):
        raise ValueError(
            f"resource targets must have shape [num_tasks, {RESOURCE_DIM}]"
        )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("edge_index must contain integer node indices")
    for name, value in (
        ("features", features),
        ("targets", targets),
    ):
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError(f"{name} must contain finite, non-negative values")
    if edge_index.numel() and (
        (edge_index < 0).any() or (edge_index >= features.shape[0]).any()
    ):
        raise ValueError("edge_index contains an out-of-range task index")
    return features, edge_index, targets


def task_feature_tensor(predictions: Sequence[object]) -> torch.Tensor:
    """Extract the four ordered regressor outputs into the GNN input matrix."""
    if not predictions:
        raise ValueError("at least one task prediction is required")
    rows = [
        [
            prediction.target_exec_time,
            prediction.target_cpu_req,
            prediction.target_gpu_req,
            prediction.target_ram_req,
        ]
        for prediction in predictions
    ]
    features = torch.tensor(rows, dtype=torch.float32)
    if not torch.isfinite(features).all():
        raise ValueError("predicted task features must contain only finite values")
    return features


def task_gnn_feature_tensor(
    tasks: Sequence[object],
    contexts: Sequence[object],
    predictions: Sequence[object],
) -> torch.Tensor:
    """Join regressor outputs with normalized raw metadata for GNN learning.

    Columns are four baseline outputs followed by eight normalized input
    properties: file size, disk size, document type, bandwidth, latency,
    available RAM, and CPU/GPU hints.
    """
    if not tasks or len(tasks) != len(contexts) or len(tasks) != len(predictions):
        raise ValueError("tasks, contexts and predictions must have matching non-zero lengths")
    rows = []
    for task, context, prediction in zip(tasks, contexts, predictions):
        baseline = [
            prediction.target_exec_time,
            prediction.target_cpu_req,
            prediction.target_gpu_req,
            prediction.target_ram_req,
        ]
        raw_features = [
            task.file_size / 16384,
            task.disk_size / 32768,
            {"text": 0.0, "image": 1 / 3, "video": 2 / 3, "archive": 1.0}[
                task.doc_type
            ],
            context.available_bandwidth / 1000,
            context.latency / 120,
            context.available_ram / 64,
            task.baseline_cpu_req / 4,
            task.baseline_gpu_req / 2,
        ]
        rows.append(baseline + raw_features)
    features = torch.tensor(rows, dtype=torch.float32)
    if not torch.isfinite(features).all() or (features < 0).any():
        raise ValueError("GNN input features must be finite and non-negative")
    return features


__all__ = [
    "TaskGraphEncoder",
    "build_dependency_edges",
    "task_gnn_feature_tensor",
    "task_feature_tensor",
]
