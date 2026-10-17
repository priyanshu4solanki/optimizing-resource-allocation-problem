"""Train and evaluate dataset-specific Gradient Boosting models."""

import argparse
import csv
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import joblib
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


CATEGORICAL_FEATURES = (
    "data_source",
    "task_type",
    "job_priority",
    "predicted_load_condition",
)
NUMERIC_FEATURES = ("number_of_active_users",)
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REGRESSION_TARGETS = {
    "execution_time_sec": "target_execution_time_sec",
    "cpu_utilization_pct": "target_cpu_utilization_pct",
    "cpu_allocation_pct": "target_cpu_allocation_pct",
    "ram_allocated_mb": "target_ram_allocated_mb",
    "ram_used_mb": "target_ram_used_mb",
}
GPU_TARGET = "gpu_required_pseudo_label"
RANDOM_STATE = 2026


def _pipeline(estimator: object) -> Pipeline:
    return Pipeline([("features", DictVectorizer()), ("model", estimator)])


def _regressor() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=120,
        learning_rate=0.04,
        max_depth=2,
        loss="huber",
        random_state=RANDOM_STATE,
    )


def _read_and_validate(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError("dataset has no CSV header")
        frame = list(reader)
    required = set(FEATURES) | set(REGRESSION_TARGETS.values()) | {GPU_TARGET}
    missing = sorted(required.difference(reader.fieldnames))
    if missing:
        raise ValueError("dataset is missing required columns: " + ", ".join(missing))
    if len(frame) < 10:
        raise ValueError("at least 10 dataset rows are required")
    if "task_start_time" not in reader.fieldnames:
        raise ValueError(
            "task_start_time is required for chronological train/test separation"
        )
    for row_number, row in enumerate(frame, start=2):
        for feature in FEATURES:
            if not row.get(feature, "").strip():
                raise ValueError(f"row {row_number}: feature {feature!r} is empty")
        try:
            row["number_of_active_users"] = float(row["number_of_active_users"])
        except ValueError as error:
            raise ValueError(
                f"row {row_number}: number_of_active_users must be numeric"
            ) from error
        for target in REGRESSION_TARGETS.values():
            try:
                value = float(row[target])
            except ValueError as error:
                raise ValueError(
                    f"row {row_number}: target {target!r} must be numeric"
                ) from error
            if not np.isfinite(value):
                raise ValueError(f"row {row_number}: target {target!r} must be finite")
            row[target] = value
        try:
            gpu_value = int(row[GPU_TARGET])
        except ValueError as error:
            raise ValueError(
                f"row {row_number}: GPU pseudo-label must be binary"
            ) from error
        if gpu_value not in (0, 1):
            raise ValueError(f"row {row_number}: GPU pseudo-label must be binary")
        row[GPU_TARGET] = gpu_value
        try:
            row["_split_time"] = datetime.fromisoformat(row["task_start_time"])
        except ValueError as error:
            raise ValueError(
                f"row {row_number}: invalid task_start_time"
            ) from error
    return sorted(frame, key=lambda row: row["_split_time"])


def _feature_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            feature: row[feature]
            for feature in FEATURES
        }
        for row in rows
    ]


def train_dataset_models(
    input_path: Path, artifact_path: Path, metrics_path: Path
) -> Dict[str, object]:
    frame = _read_and_validate(input_path)
    split_index = int(len(frame) * 0.8)
    if split_index <= 0 or split_index >= len(frame):
        raise ValueError("chronological split must leave rows for train and test")
    train = frame[:split_index]
    test = frame[split_index:]
    x_train = _feature_rows(train)
    x_test = _feature_rows(test)

    regression_metrics: Dict[str, object] = {}
    final_models: Dict[str, Pipeline] = {}
    for name, target_column in REGRESSION_TARGETS.items():
        y_train = np.asarray([row[target_column] for row in train], dtype=float)
        y_test = np.asarray([row[target_column] for row in test], dtype=float)
        evaluation_model = _pipeline(_regressor())
        evaluation_model.fit(x_train, y_train)
        prediction = evaluation_model.predict(x_test)
        mean_baseline = np.full_like(y_test, fill_value=float(np.mean(y_train)))
        regression_metrics[name] = {
            "target_column": target_column,
            "unit": {
                "execution_time_sec": "seconds",
                "cpu_utilization_pct": "percent",
                "cpu_allocation_pct": "percent of unknown host/VM capacity",
                "ram_allocated_mb": "MB; Memory_Share * VM_Total_Memory_Capacity",
                "ram_used_mb": "MB; observed Memory_Consumption",
            }[name],
            "test_mae": float(mean_absolute_error(y_test, prediction)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
            "test_r2": float(r2_score(y_test, prediction)),
            "mean_baseline_mae": float(mean_absolute_error(y_test, mean_baseline)),
            "mae_improvement_vs_mean_baseline_fraction": float(
                1.0
                - mean_absolute_error(y_test, prediction)
                / max(mean_absolute_error(y_test, mean_baseline), 1e-12)
            ),
        }
        final_model = _pipeline(_regressor())
        final_model.fit(
            _feature_rows(frame),
            np.asarray([row[target_column] for row in frame], dtype=float),
        )
        final_models[name] = final_model

    gpu_train = np.asarray([row[GPU_TARGET] for row in train], dtype=int)
    gpu_test = np.asarray([row[GPU_TARGET] for row in test], dtype=int)
    gpu_model = _pipeline(
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.04,
            max_depth=2,
            random_state=RANDOM_STATE,
        )
    )
    gpu_model.fit(x_train, gpu_train)
    gpu_prediction = gpu_model.predict(x_test).astype(int)
    all_negative = np.zeros_like(gpu_test)
    gpu_metrics = {
        "target_column": GPU_TARGET,
        "label_provenance": "synthetic heuristic; not measured GPU need",
        "test_positive_fraction": float(np.mean(gpu_test)),
        "test_accuracy": float(accuracy_score(gpu_test, gpu_prediction)),
        "test_balanced_accuracy": float(
            balanced_accuracy_score(gpu_test, gpu_prediction)
        ),
        "test_precision_positive": float(
            precision_score(gpu_test, gpu_prediction, zero_division=0)
        ),
        "test_recall_positive": float(
            recall_score(gpu_test, gpu_prediction, zero_division=0)
        ),
        "test_f1_positive": float(
            f1_score(gpu_test, gpu_prediction, zero_division=0)
        ),
        "always_zero_baseline_accuracy": float(
            accuracy_score(gpu_test, all_negative)
        ),
        "always_zero_baseline_balanced_accuracy": float(
            balanced_accuracy_score(gpu_test, all_negative)
        ),
        "deployment_usable": False,
    }
    gpu_final_model = _pipeline(
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.04,
            max_depth=2,
            random_state=RANDOM_STATE,
        )
    )
    gpu_final_model.fit(
        _feature_rows(frame),
        np.asarray([row[GPU_TARGET] for row in frame], dtype=int),
    )
    final_models["gpu_pseudo_label_classifier"] = gpu_final_model

    metrics: Dict[str, object] = {
        "input_dataset": str(input_path),
        "row_count": len(frame),
        "train_rows": len(train),
        "test_rows": len(test),
        "split_method": "chronological 80/20 split by task_start_time",
        "train_time_range": [
            min(row["_split_time"] for row in train).isoformat(),
            max(row["_split_time"] for row in train).isoformat(),
        ],
        "test_time_range": [
            min(row["_split_time"] for row in test).isoformat(),
            max(row["_split_time"] for row in test).isoformat(),
        ],
        "input_features": list(FEATURES),
        "excluded_post_execution_or_leakage_columns": [
            "task_size_proxy_mb",
            "observed_network_bandwidth_utilization_mbps",
            "target_execution_time_sec",
            "target_cpu_utilization_pct",
            "target_cpu_allocation_pct",
            "target_ram_allocated_mb",
            "target_ram_used_mb",
            "target_waiting_time_ms",
            "actual_host_id",
            "predicted_host_id",
            "host_selection_correct",
        ],
        "regression": regression_metrics,
        "gpu_pseudo_label_classifier": gpu_metrics,
        "limitations": [
            "The dataset does not contain true task file size or measured GPU use.",
            "Memory consumption and CPU usage are outcomes; they are targets, not inputs.",
            "RAM allocation is derived from Memory_Share and VM capacity; it is not "
            "equivalent to memory consumed.",
            "CPU percent and CPU-share labels lack host core-count and precise "
            "allocation semantics, so they cannot be mapped directly to scheduler CPU units.",
            "GPU classification only learns the synthetic heuristic label and is "
            "not suitable for live GPU placement.",
            "Models use the dataset's features and target units and are not wired "
            "into the current live hub feature contract.",
            "Dataset provenance and whether values are real measurements should be "
            "verified before operational use.",
        ],
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "models": final_models,
            "features": list(FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "numeric_features": list(NUMERIC_FEATURES),
            "regression_targets": REGRESSION_TARGETS,
            "gpu_target": GPU_TARGET,
            "training_rows": len(frame),
            "training_time_range": metrics["train_time_range"],
            "dataset_time_range": [
                min(row["_split_time"] for row in frame).isoformat(),
                max(row["_split_time"] for row in frame).isoformat(),
            ],
            "gpu_model_is_pseudo_label_only": True,
        },
        artifact_path,
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    args = parser.parse_args()
    metrics = train_dataset_models(args.input, args.artifact, args.metrics)
    print(f"Saved dataset-specific models to {args.artifact}")
    print(f"Saved held-out evaluation metrics to {args.metrics}")
    print(json.dumps(metrics["regression"], indent=2))
    print("GPU pseudo-label evaluation:")
    print(json.dumps(metrics["gpu_pseudo_label_classifier"], indent=2))


if __name__ == "__main__":
    main()
