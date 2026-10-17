"""Clean the supplied cloud task CSV and add explicitly synthetic GPU labels."""

import argparse
import csv
import json
import math
import statistics
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


SOURCE_COLUMNS = {
    "job_id": "Job_ID",
    "task_start_time": "Task_Start_Time",
    "cpu_utilization_pct": "CPU_Utilization (%)",
    "memory_consumption_mb": "Memory_Consumption (MB)",
    "execution_time_ms": "Task_Execution_Time (ms)",
    "waiting_time_ms": "Task_Waiting_Time (ms)",
    "data_source": "Data_Source",
    "active_users": "Number_of_Active_Users",
    "network_bandwidth_utilization_mbps": "Network_Bandwidth_Utilization (Mbps)",
    "job_priority": "Job_Priority",
    "task_type": "Task_Type",
    "predicted_load_condition": "Predicted_Load_Condition",
    "cpu_share_pct": "CPU_Share (%)",
    "memory_share_pct": "Memory_Share (%)",
    "vm_memory_capacity_mb": "VM_Total_Memory_Capacity (MB)",
    "actual_host_id": "Actual_Host_ID",
    "predicted_host_id": "Predicted_Host_ID",
    "host_selection_correct": "Host_Selection_Correct",
}

OUTPUT_COLUMNS = (
    "job_id",
    "task_start_time",
    "data_source",
    "task_type",
    "job_priority",
    "number_of_active_users",
    "predicted_load_condition",
    "observed_network_bandwidth_utilization_mbps",
    "task_size_proxy_mb",
    "target_execution_time_sec",
    "target_cpu_utilization_pct",
    "target_cpu_allocation_pct",
    "target_ram_allocated_mb",
    "target_ram_used_mb",
    "target_waiting_time_ms",
    "actual_host_id",
    "predicted_host_id",
    "host_selection_correct",
    "gpu_required_pseudo_label",
    "gpu_requirement_units_pseudo",
)


def _load_rows(archive_path: Path, member: str = "") -> Tuple[str, List[Dict[str, str]]]:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            csv_members = [
                item.filename
                for item in archive.infolist()
                if item.filename.lower().endswith(".csv")
            ]
            if member:
                if member not in csv_members:
                    raise ValueError(f"CSV member {member!r} not found in archive")
                selected = member
            elif len(csv_members) == 1:
                selected = csv_members[0]
            else:
                raise ValueError(
                    "archive must contain exactly one CSV or specify --member"
                )
            with archive.open(selected) as binary:
                import io

                reader = csv.DictReader(
                    io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
                )
                return selected, list(reader)

    with archive_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        return archive_path.name, list(reader)


def _number(row: Dict[str, str], column: str, row_number: int) -> float:
    raw = row[column].strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"row {row_number}: {column!r} must be numeric, got {raw!r}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(f"row {row_number}: {column!r} must be finite")
    return value


def _quantile_75(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot calculate GPU thresholds from an empty dataset")
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=4, method="inclusive")[2]


def clean_dataset(
    rows: Sequence[Dict[str, str]],
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not rows:
        raise ValueError("input dataset contains no rows")
    required = set(SOURCE_COLUMNS.values())
    missing = sorted(required.difference(rows[0].keys()))
    if missing:
        raise ValueError("input dataset is missing required columns: " + ", ".join(missing))

    parsed: List[Dict[str, object]] = []
    cpu_values = []
    size_values = []
    for index, row in enumerate(rows, start=2):
        start_text = row[SOURCE_COLUMNS["task_start_time"]].strip()
        try:
            start_time = datetime.strptime(start_text, "%d-%m-%Y %H:%M")
        except ValueError as error:
            raise ValueError(
                f"row {index}: invalid Task_Start_Time {start_text!r}"
            ) from error

        values = {
            name: _number(row, source_column, index)
            for name, source_column in SOURCE_COLUMNS.items()
            if name
            in {
                "cpu_utilization_pct",
                "memory_consumption_mb",
                "execution_time_ms",
                "waiting_time_ms",
                "active_users",
                "network_bandwidth_utilization_mbps",
                "cpu_share_pct",
                "memory_share_pct",
                "vm_memory_capacity_mb",
                "host_selection_correct",
            }
        }
        if values["execution_time_ms"] < 0 or values["memory_consumption_mb"] < 0:
            raise ValueError(f"row {index}: duration and memory use must be non-negative")
        if not 0 <= values["cpu_utilization_pct"] <= 100:
            raise ValueError(f"row {index}: CPU utilization must be in [0, 100]")
        if not 0 <= values["cpu_share_pct"] <= 100:
            raise ValueError(f"row {index}: CPU share must be in [0, 100]")
        if not 0 <= values["memory_share_pct"] <= 100:
            raise ValueError(f"row {index}: memory share must be in [0, 100]")

        task_size_proxy = values["memory_consumption_mb"]
        cpu_values.append(values["cpu_utilization_pct"])
        size_values.append(task_size_proxy)
        parsed.append(
            {
                "job_id": row[SOURCE_COLUMNS["job_id"]].strip(),
                "task_start_time": start_time.isoformat(timespec="minutes"),
                "data_source": row[SOURCE_COLUMNS["data_source"]].strip(),
                "task_type": row[SOURCE_COLUMNS["task_type"]].strip(),
                "job_priority": row[SOURCE_COLUMNS["job_priority"]].strip(),
                "number_of_active_users": int(values["active_users"]),
                "predicted_load_condition": row[
                    SOURCE_COLUMNS["predicted_load_condition"]
                ].strip(),
                "observed_network_bandwidth_utilization_mbps": values[
                    "network_bandwidth_utilization_mbps"
                ],
                "task_size_proxy_mb": task_size_proxy,
                "target_execution_time_sec": values["execution_time_ms"] / 1000.0,
                "target_cpu_utilization_pct": values["cpu_utilization_pct"],
                "target_cpu_allocation_pct": values["cpu_share_pct"],
                "target_ram_allocated_mb": (
                    values["memory_share_pct"]
                    / 100.0
                    * values["vm_memory_capacity_mb"]
                ),
                "target_ram_used_mb": values["memory_consumption_mb"],
                "target_waiting_time_ms": values["waiting_time_ms"],
                "actual_host_id": row[SOURCE_COLUMNS["actual_host_id"]].strip(),
                "predicted_host_id": row[SOURCE_COLUMNS["predicted_host_id"]].strip(),
                "host_selection_correct": int(values["host_selection_correct"]),
            }
        )

    cpu_threshold = _quantile_75(cpu_values)
    size_threshold = _quantile_75(size_values)
    pseudo_gpu_count = 0
    for row in parsed:
        use_gpu = (
            row["target_cpu_utilization_pct"] >= cpu_threshold
            and row["task_size_proxy_mb"] >= size_threshold
        )
        row["gpu_required_pseudo_label"] = int(use_gpu)
        row["gpu_requirement_units_pseudo"] = 1.0 if use_gpu else 0.0
        pseudo_gpu_count += int(use_gpu)

    metadata: Dict[str, object] = {
        "rows": len(parsed),
        "source_columns": len(rows[0]),
        "output_columns": list(OUTPUT_COLUMNS),
        "cpu_high_threshold_pct_q75": cpu_threshold,
        "task_size_proxy_threshold_mb_q75": size_threshold,
        "gpu_pseudo_positive_rows": pseudo_gpu_count,
        "gpu_pseudo_positive_fraction": pseudo_gpu_count / len(parsed),
        "gpu_pseudo_label_rule": (
            "1 GPU unit if observed CPU utilization and observed memory "
            "consumption are both at or above their dataset 75th percentiles; "
            "otherwise 0"
        ),
        "warnings": [
            "GPU labels are heuristic pseudo-labels, not measured GPU need or use.",
            "Memory_Consumption is post-execution RAM use, not input file size. "
            "task_size_proxy_mb is retained for analysis and GPU-label construction "
            "but must not be used as a pre-execution model input.",
            "CPU utilization and other target_* columns are observed outcomes and "
            "must not be used as pre-execution model inputs.",
            "target_ram_allocated_mb is Memory_Share percent multiplied by VM "
            "memory capacity; it represents allocated RAM, not necessarily used RAM.",
            "The dataset has no task file-size field and no measured GPU telemetry.",
        ],
    }
    return parsed, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="input CSV or ZIP")
    parser.add_argument("--output", required=True, type=Path, help="cleaned CSV path")
    parser.add_argument(
        "--member", default="", help="CSV member name when the ZIP has multiple CSVs"
    )
    args = parser.parse_args()

    source_member, rows = _load_rows(args.input, args.member)
    cleaned, metadata = clean_dataset(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(cleaned)

    metadata["source_archive"] = args.input.name
    metadata["source_member"] = source_member
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(cleaned)} rows and {len(OUTPUT_COLUMNS)} columns to {args.output}"
    )
    print(f"Wrote derivation details to {metadata_path}")
    print(
        "GPU pseudo-label thresholds: "
        f"CPU >= {metadata['cpu_high_threshold_pct_q75']:.2f}%, "
        f"task-size proxy >= {metadata['task_size_proxy_threshold_mb_q75']:.2f} MB; "
        f"{metadata['gpu_pseudo_positive_rows']} positive labels"
    )


if __name__ == "__main__":
    main()
