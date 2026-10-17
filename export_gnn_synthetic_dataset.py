"""Export the exact default synthetic examples and features used by the GNN demo."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from gnn_encoder import task_gnn_feature_tensor
from task_predictors import (
    DOC_TYPE_CODES,
    TaskResourcePredictor,
    generate_synthetic_training_data,
    synthetic_resource_targets,
)


def _synthetic_parent_indices(sample_count: int, seed: int) -> Dict[int, List[int]]:
    """Mirror main_engine._synthetic_dag_edges without requiring graph libraries."""
    parents: Dict[int, List[int]] = {}
    for index in range(sample_count):
        parents[index] = [] if index % 5 == 0 else [index - 1]
    if seed % 2:
        for index in range(4, sample_count, 7):
            parents[index] = [index - 2]
    return parents


def export_dataset(output_path: Path, seed: int = 7, sample_count: int = 600) -> None:
    if sample_count < 10:
        raise ValueError("sample_count must be at least 10")
    split_index = int(sample_count * 0.8)
    if split_index <= 0 or split_index >= sample_count:
        raise ValueError("dataset split must leave training and validation samples")

    tasks, contexts = generate_synthetic_training_data(
        sample_count=sample_count, seed=seed
    )
    predictor = TaskResourcePredictor(random_state=seed)
    predictor.fit_synthetic(tasks[:split_index], contexts[:split_index])
    train_predictions = predictor.predict(tasks[:split_index], contexts[:split_index])
    validation_predictions = predictor.predict(
        tasks[split_index:], contexts[split_index:]
    )
    train_features = task_gnn_feature_tensor(
        tasks[:split_index], contexts[:split_index], train_predictions
    )
    validation_features = task_gnn_feature_tensor(
        tasks[split_index:], contexts[split_index:], validation_predictions
    )
    targets = synthetic_resource_targets(tasks, contexts)
    train_parents = _synthetic_parent_indices(split_index, seed)
    validation_count = sample_count - split_index
    validation_parents = _synthetic_parent_indices(validation_count, seed + 1)

    columns = [
        "sample_index",
        "task_id",
        "split",
        "graph_local_node_id",
        "synthetic_parent_node_ids",
        "file_size_kib",
        "disk_size_kib",
        "doc_type",
        "doc_type_code",
        "baseline_cpu_hint",
        "baseline_gpu_hint",
        "available_bandwidth",
        "latency",
        "available_ram",
        "predictor_target_exec_time",
        "predictor_target_cpu_req",
        "predictor_target_gpu_req",
        "predictor_target_ram_req",
    ]
    columns.extend(f"gnn_input_{index:02d}" for index in range(12))
    columns.extend(
        [
            "synthetic_target_exec_time",
            "synthetic_target_cpu_req",
            "synthetic_target_gpu_req",
            "synthetic_target_ram_req",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for index, (task, context, target) in enumerate(
            zip(tasks, contexts, targets)
        ):
            is_train = index < split_index
            if is_train:
                local_index = index
                graph_parents = train_parents[local_index]
                prediction = train_predictions[local_index]
                features = train_features[local_index]
                split = "train"
                graph_prefix = "train"
            else:
                local_index = index - split_index
                graph_parents = validation_parents[local_index]
                prediction = validation_predictions[local_index]
                features = validation_features[local_index]
                split = "validation"
                graph_prefix = "validation"

            row = {
                "sample_index": index,
                "task_id": task.task_id,
                "split": split,
                "graph_local_node_id": local_index,
                "synthetic_parent_node_ids": json.dumps(
                    [f"{graph_prefix}-{parent}" for parent in graph_parents]
                ),
                "file_size_kib": task.file_size,
                "disk_size_kib": task.disk_size,
                "doc_type": task.doc_type,
                "doc_type_code": DOC_TYPE_CODES[task.doc_type],
                "baseline_cpu_hint": task.baseline_cpu_req,
                "baseline_gpu_hint": task.baseline_gpu_req,
                "available_bandwidth": context.available_bandwidth,
                "latency": context.latency,
                "available_ram": context.available_ram,
                "predictor_target_exec_time": prediction.target_exec_time,
                "predictor_target_cpu_req": prediction.target_cpu_req,
                "predictor_target_gpu_req": prediction.target_gpu_req,
                "predictor_target_ram_req": prediction.target_ram_req,
                "synthetic_target_exec_time": target["target_exec_time"],
                "synthetic_target_cpu_req": target["target_cpu_req"],
                "synthetic_target_gpu_req": target["target_gpu_req"],
                "synthetic_target_ram_req": target["target_ram_req"],
            }
            row.update(
                {
                    f"gnn_input_{feature_index:02d}": float(value)
                    for feature_index, value in enumerate(features.tolist())
                }
            )
            writer.writerow(row)

    metadata = {
        "purpose": "Reproduction/export of the default GNN offline demo data",
        "seed": seed,
        "sample_count": sample_count,
        "train_count": split_index,
        "validation_count": sample_count - split_index,
        "split": "First 80% of generated rows train; final 20% validation",
        "training_predictor": (
            "TaskResourcePredictor(random_state=seed), fit_synthetic on the "
            "training rows only; its estimates plus normalized raw inputs form "
            "the 12 GNN inputs"
        ),
        "graph": (
            "Artificial short-chain links from main_engine._synthetic_dag_edges; "
            "training graph seed=seed, validation graph seed=seed+1. These edges "
            "are fabricated and are not real workflow dependencies."
        ),
        "synthetic_targets": (
            "Generated by synthetic_resource_targets / _synthetic_targets. "
            "They are rule-generated labels, not measurements."
        ),
        "gnn_input_columns": [
            "0 target execution time estimate",
            "1 target CPU estimate",
            "2 target GPU estimate",
            "3 target RAM estimate",
            "4 file_size / 16384",
            "5 disk_size / 32768",
            "6 normalized document-type code",
            "7 available_bandwidth / 1000",
            "8 latency / 120",
            "9 available_ram / 64",
            "10 baseline_cpu_req / 4",
            "11 baseline_gpu_req / 2",
        ],
        "warning": (
            "This synthetic dataset only documents/reproduces the demo training "
            "inputs. It is not real workload data and should not be used as "
            "evidence of production prediction accuracy."
        ),
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {sample_count} samples to {output_path}")
    print(f"Wrote dataset notes to {output_path.with_suffix('.metadata.json')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gnn_synthetic_training_dataset.csv"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=600)
    args = parser.parse_args()
    export_dataset(args.output, seed=args.seed, sample_count=args.samples)


if __name__ == "__main__":
    main()
