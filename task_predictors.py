"""Gradient-boosted predictors for task resource and execution-time estimates."""

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


FEATURE_NAMES = (
    "file_size",
    "disk_size",
    "doc_type_code",
    "available_bandwidth",
    "latency",
    "available_ram",
    "baseline_cpu_req",
    "baseline_gpu_req",
)
TARGET_NAMES = (
    "target_exec_time",
    "target_cpu_req",
    "target_gpu_req",
    "target_ram_req",
)
DOC_TYPE_CODES = {"text": 0.0, "image": 1.0, "video": 2.0, "archive": 3.0}


@dataclass(frozen=True)
class InfrastructureContext:
    available_bandwidth: float
    latency: float
    available_ram: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class TaskInput:
    task_id: str
    file_size: float
    disk_size: float
    doc_type: str
    parents: Tuple[str, ...] = ()
    baseline_cpu_req: float = 0.0
    baseline_gpu_req: float = 0.0

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must not be empty")
        if self.doc_type not in DOC_TYPE_CODES:
            raise ValueError(
                f"unsupported doc_type {self.doc_type!r}; "
                f"expected one of {tuple(DOC_TYPE_CODES)}"
            )
        for name in (
            "file_size",
            "disk_size",
            "baseline_cpu_req",
            "baseline_gpu_req",
        ):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class PredictedTask:
    task_id: str
    parents: Tuple[str, ...]
    target_exec_time: float
    target_cpu_req: float
    target_gpu_req: float
    target_ram_req: float


def _feature_row(task: TaskInput, infrastructure: InfrastructureContext) -> List[float]:
    return [
        task.file_size,
        task.disk_size,
        DOC_TYPE_CODES[task.doc_type],
        infrastructure.available_bandwidth,
        infrastructure.latency,
        infrastructure.available_ram,
        task.baseline_cpu_req,
        task.baseline_gpu_req,
    ]


class TaskResourcePredictor:
    """Four independent GradientBoostingRegressor models with explicit features."""

    def __init__(self, random_state: int = 7) -> None:
        self.random_state = random_state
        self.models: Dict[str, GradientBoostingRegressor] = {}

    @property
    def is_fitted(self) -> bool:
        return set(self.models) == set(TARGET_NAMES)

    def fit(
        self,
        tasks: Sequence[TaskInput],
        contexts: Sequence[InfrastructureContext],
        targets: Sequence[Mapping[str, float]],
    ) -> None:
        if len(tasks) != len(contexts):
            raise ValueError("tasks and contexts must have the same length")
        if len(tasks) < 2:
            raise ValueError("at least two training samples are required")
        if len(targets) != len(tasks):
            raise ValueError("targets must contain one resource-label row per task")
        if any(set(target) != set(TARGET_NAMES) for target in targets):
            raise ValueError(
                "each target row must contain exactly "
                + ", ".join(TARGET_NAMES)
            )
        inputs = np.asarray(
            [_feature_row(task, context) for task, context in zip(tasks, contexts)],
            dtype=np.float64,
        )
        target_matrix = np.asarray(
            [
                [target[name] for name in TARGET_NAMES]
                for target in targets
            ],
            dtype=np.float64,
        )
        if not np.isfinite(target_matrix).all() or (target_matrix < 0).any():
            raise ValueError("training targets must be finite and non-negative")
        self.models = {}
        for index, target_name in enumerate(TARGET_NAMES):
            model = GradientBoostingRegressor(
                n_estimators=80,
                max_depth=2,
                loss="huber",
                random_state=self.random_state + index,
            )
            model.fit(inputs, target_matrix[:, index])
            self.models[target_name] = model

    def fit_synthetic(
        self, tasks: Sequence[TaskInput], contexts: Sequence[InfrastructureContext]
    ) -> None:
        """Train on the demo's generated labels; never use for production artifacts."""
        self.fit(tasks, contexts, synthetic_resource_targets(tasks, contexts))

    def predict(
        self, tasks: Sequence[TaskInput], contexts: Sequence[InfrastructureContext]
    ) -> List[PredictedTask]:
        if not self.is_fitted:
            raise RuntimeError("predictor has not been fitted")
        if len(tasks) != len(contexts):
            raise ValueError("tasks and contexts must have the same length")
        if not tasks:
            return []
        inputs = np.asarray(
            [_feature_row(task, context) for task, context in zip(tasks, contexts)],
            dtype=np.float64,
        )
        outputs = np.column_stack(
            [self.models[target_name].predict(inputs) for target_name in TARGET_NAMES]
        )
        predictions: List[PredictedTask] = []
        for task, output in zip(tasks, outputs):
            values = np.maximum(output, 0.0)
            values[1] = max(values[1], task.baseline_cpu_req)
            values[2] = max(values[2], task.baseline_gpu_req)
            predictions.append(
                PredictedTask(
                    task_id=task.task_id,
                    parents=task.parents,
                    target_exec_time=float(values[0]),
                    target_cpu_req=float(values[1]),
                    target_gpu_req=float(values[2]),
                    target_ram_req=float(values[3]),
                )
            )
        return predictions

    def feature_row(
        self, task: TaskInput, infrastructure: InfrastructureContext
    ) -> Tuple[float, ...]:
        """Return the named, ordered numeric features used by this predictor."""
        return tuple(_feature_row(task, infrastructure))


def _synthetic_targets(
    task: TaskInput, infrastructure: InfrastructureContext
) -> Tuple[float, float, float, float]:
    """Deterministic workload rules used solely to generate demo training labels."""
    factor = {"text": 0.75, "image": 1.25, "video": 2.4, "archive": 1.1}[
        task.doc_type
    ]
    size_mb = task.file_size / 1024.0
    disk_mb = task.disk_size / 1024.0
    bandwidth = max(infrastructure.available_bandwidth, 1.0)
    execution_time = (
        0.5
        + factor * (0.15 + size_mb * 0.08 + disk_mb * 0.025)
        + infrastructure.latency * 0.015
        + size_mb / bandwidth * 0.2
    )
    cpu = max(task.baseline_cpu_req, 0.15 + factor * (0.12 + size_mb * 0.018))
    gpu = max(task.baseline_gpu_req, factor * (0.025 + size_mb * 0.006))
    ram = max(
        0.1 + disk_mb * 0.012 + size_mb * (0.035 if task.doc_type != "video" else 0.08),
        task.baseline_cpu_req * 0.35,
    )
    ram = min(ram, max(infrastructure.available_ram, 0.1))
    return execution_time, cpu, gpu, ram


def generate_synthetic_training_data(
    sample_count: int = 600, seed: int = 7
) -> Tuple[List[TaskInput], List[InfrastructureContext]]:
    """Generate deterministic labeled demo samples; not a substitute for telemetry."""
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    rng = np.random.default_rng(seed)
    tasks: List[TaskInput] = []
    contexts: List[InfrastructureContext] = []
    doc_types = tuple(DOC_TYPE_CODES)
    for index in range(sample_count):
        doc_type = doc_types[int(rng.integers(0, len(doc_types)))]
        tasks.append(
            TaskInput(
                task_id=f"train-{index}",
                file_size=float(rng.uniform(64, 16384)),
                disk_size=float(rng.uniform(128, 32768)),
                doc_type=doc_type,
                baseline_cpu_req=float(rng.uniform(0, 4)),
                baseline_gpu_req=float(rng.uniform(0, 2)),
            )
        )
        contexts.append(
            InfrastructureContext(
                available_bandwidth=float(rng.uniform(10, 1000)),
                latency=float(rng.uniform(1, 120)),
                available_ram=float(rng.uniform(1, 64)),
            )
        )
    return tasks, contexts


def synthetic_resource_targets(
    tasks: Sequence[TaskInput], contexts: Sequence[InfrastructureContext]
) -> List[Dict[str, float]]:
    """Return deterministic synthetic labels for demos and model smoke tests."""
    if len(tasks) != len(contexts):
        raise ValueError("tasks and contexts must have the same length")
    return [
        dict(zip(TARGET_NAMES, _synthetic_targets(task, context)))
        for task, context in zip(tasks, contexts)
    ]


__all__ = [
    "DOC_TYPE_CODES",
    "FEATURE_NAMES",
    "TARGET_NAMES",
    "InfrastructureContext",
    "PredictedTask",
    "TaskInput",
    "TaskResourcePredictor",
    "generate_synthetic_training_data",
    "synthetic_resource_targets",
]
