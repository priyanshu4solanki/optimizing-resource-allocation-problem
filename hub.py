"""FastAPI telemetry and live task-allocation hub."""

import json
import hashlib
import asyncio
import logging
import os
import re
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from ast_dependency_detector import (
    DependencyAnalysisError,
    detect_import_dependencies,
)
from gnn_encoder import (
    TaskGraphEncoder,
    build_dependency_edges,
    task_gnn_feature_tensor,
)
from marl_scheduler import MARLScheduler, SchedulerExperience
from marl_scheduler_bridge import build_joint_state
from mappo_scheduler import MAPPOScheduler, load_scheduler_checkpoint
from model_persistence import load_gnn_encoder, load_predictor
from task_predictors import (
    InfrastructureContext,
    PredictedTask,
    TaskInput,
    TaskResourcePredictor,
)

LOGGER = logging.getLogger(__name__)
TASK_SCRIPT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\.py$")
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
UPLOAD_SCAN_INTERVAL_SEC = 0.1
UPLOAD_STABILITY_SEC = 0.25
SCHEDULE_RETRY_LOG_INTERVAL_SEC = 5.0


class Metrics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cpu_percent: float = Field(ge=0, le=100)
    ram_percent: float = Field(ge=0, le=100)
    ram_free_gb: float = Field(ge=0)
    ram_total_gb: float = Field(gt=0)
    active_tasks: int = Field(default=0, ge=0)
    net_tx_kbps: float = Field(default=0, ge=0)
    net_rx_kbps: float = Field(default=0, ge=0)
    rtt_ms: float = Field(default=0, ge=0)
    cores_logical: int = Field(gt=0)
    gpu_count: float = Field(default=0, ge=0)
    gpu_percent: float = Field(default=0, ge=0, le=100)
    cost_per_second: float = Field(default=0.03, ge=0)


class TelemetryPacket(BaseModel):
    model_config = ConfigDict(extra="ignore")

    node_id: str = Field(min_length=1, max_length=64)
    ip_address: str = Field(min_length=1, max_length=64)
    timestamp: float
    status: Literal["FREE", "OCCUPIED", "HIGH_LOAD"]
    metrics: Metrics


class SubmittedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    script_name: str = Field(min_length=1, max_length=128)
    file_size: float = Field(ge=0)
    disk_size: float = Field(ge=0)
    doc_type: Literal["text", "image", "video", "archive"]
    parents: List[str] = Field(default_factory=list)
    baseline_cpu_req: float = Field(default=0, ge=0)
    baseline_gpu_req: float = Field(default=0, ge=0)


class TaskBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: List[SubmittedTask] = Field(min_length=1, max_length=100)


@dataclass(frozen=True)
class ModelBundle:
    predictor: TaskResourcePredictor
    encoder: TaskGraphEncoder
    scheduler: Union[MARLScheduler, MAPPOScheduler]


def load_model_bundle_from_environment() -> Optional[ModelBundle]:
    """Load configured model paths, or the local trained checkpoints by default."""
    names = ("MARL_PREDICTOR_PATH", "MARL_GNN_PATH", "MARL_SCHEDULER_PATH")
    configured_paths = [os.environ.get(name) for name in names]
    if not any(configured_paths):
        project_root = Path(__file__).resolve().parent
        default_paths = (
            project_root / "models" / "task_predictor.joblib",
            project_root / "models" / "task_gnn.pt",
            project_root / "results" / "mappo_reference_1000" / "mappo_scheduler.pt",
        )
        if not all(path.is_file() for path in default_paths):
            return None
        paths = [str(path) for path in default_paths]
    elif not all(configured_paths):
        raise RuntimeError(
            "Set MARL_PREDICTOR_PATH, MARL_GNN_PATH, and MARL_SCHEDULER_PATH together"
        )
    else:
        paths = [str(path) for path in configured_paths]
    device = torch.device(os.environ.get("MARL_DEVICE", "cpu"))
    predictor = load_predictor(paths[0])
    encoder = load_gnn_encoder(paths[1], device=device)
    scheduler = load_scheduler_checkpoint(paths[2], device=device)
    if scheduler.input_dim != encoder.embedding_dim + 7:
        raise ValueError(
            "scheduler checkpoint input_dim must equal GNN embedding_dim + 7"
        )
    return ModelBundle(predictor, encoder, scheduler)


class HubRuntime:
    def __init__(
        self,
        *,
        node_ids: Optional[Sequence[str]] = None,
        models: Optional[ModelBundle] = None,
        task_directory: Optional[Path] = None,
        heartbeat_timeout_sec: float = 3.0,
        api_token: Optional[str] = None,
        upload_stability_sec: float = UPLOAD_STABILITY_SEC,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
    ) -> None:
        if node_ids is None:
            node_ids = tuple(
                item.strip()
                for item in os.environ.get("MARL_NODE_IDS", "S1,S2,S3").split(",")
                if item.strip()
            )
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("node_ids must be a non-empty unique sequence")
        max_upload_bytes = int(os.environ.get("MARL_MAX_UPLOAD_BYTES", max_upload_bytes))
        upload_stability_sec = float(
            os.environ.get("MARL_UPLOAD_STABILITY_SEC", upload_stability_sec)
        )
        if heartbeat_timeout_sec <= 0:
            raise ValueError("heartbeat_timeout_sec must be positive")
        if upload_stability_sec < 0 or max_upload_bytes <= 0:
            raise ValueError("upload_stability_sec must be non-negative and max_upload_bytes positive")
        self.node_ids = tuple(node_ids)
        self.models = models
        self.task_directory = Path(
            task_directory
            if task_directory is not None
            else os.environ.get("MARL_TASKS_DIR", r"\\192.168.137.1\cluster_task")
        )
        self.api_token = api_token if api_token is not None else os.environ.get("MARL_API_TOKEN")
        self.scheduler_checkpoint_path = os.environ.get("MARL_SCHEDULER_PATH")
        self.heartbeat_timeout_sec = heartbeat_timeout_sec
        self.upload_stability_sec = upload_stability_sec
        self.max_upload_bytes = max_upload_bytes
        self.model_lock = threading.RLock()
        self.experience_lock = threading.Lock()
        self.submission_lock = threading.Lock()
        self.pending_experiences: Dict[str, Tuple[Any, ...]] = {}
        self.observed_uploads: Dict[str, Tuple[int, int, float]] = {}
        self.schedule_retry_log_times: Dict[str, float] = {}
        self.cluster_state: Dict[str, Dict[str, Any]] = {
            node_id: {
                "node_id": node_id,
                "status": "OFFLINE",
                "last_seen": 0.0,
                "ip_address": "N/A",
                "metrics": {},
            }
            for node_id in self.node_ids
        }

    def ingest(self, packet: TelemetryPacket) -> None:
        if packet.node_id not in self.cluster_state:
            raise HTTPException(status_code=404, detail="unknown node_id")
        now = time.time()
        self.cluster_state[packet.node_id] = {
            "node_id": packet.node_id,
            "status": packet.status,
            "last_seen": now,
            "ip_address": packet.ip_address,
            "metrics": packet.metrics.model_dump(),
        }

    def refresh_status(self) -> None:
        now = time.time()
        for node in self.cluster_state.values():
            if node["last_seen"] == 0:
                node["status"] = "OFFLINE"
            elif now - node["last_seen"] > self.heartbeat_timeout_sec:
                node["status"] = "CRASHED"

    def network_graph(self) -> Dict[str, Any]:
        self.refresh_status()
        nodes = sorted(self.node_ids)
        features = []
        action_mask = []
        for node_id in nodes:
            entry = self.cluster_state[node_id]
            metrics = entry["metrics"]
            alive = int(entry["status"] in ("FREE", "OCCUPIED", "HIGH_LOAD"))
            action_mask.append(alive)
            features.append(
                [
                    metrics.get("cpu_percent", 0.0) / 100,
                    metrics.get("ram_percent", 0.0) / 100,
                    float(metrics.get("active_tasks", 0)),
                    float(alive),
                ]
            )
        adjacency = [[0.0 for _ in nodes] for _ in nodes]
        for source, source_id in enumerate(nodes):
            for target, target_id in enumerate(nodes):
                if source != target and action_mask[source] and action_mask[target]:
                    latency = self.cluster_state[source_id]["metrics"].get(
                        "rtt_ms", 1.0
                    )
                    adjacency[source][target] = max(latency, 0.5)
        return {
            "nodes": nodes,
            "node_features": features,
            "action_mask": action_mask,
            "adjacency_matrix": adjacency,
        }

    def submit(
        self, batch: TaskBatch, supplied_token: Optional[str]
    ) -> Dict[str, Any]:
        with self.submission_lock:
            if not self.api_token:
                raise HTTPException(
                    status_code=503,
                    detail="Task submission is disabled until MARL_API_TOKEN is configured.",
                )
            if not supplied_token or not secrets.compare_digest(
                supplied_token, self.api_token
            ):
                raise HTTPException(status_code=401, detail="invalid or missing API token")
            return self._submit_unlocked(batch)

    async def save_upload(self, request: Request, filename: str) -> Dict[str, str]:
        if not self.api_token:
            raise HTTPException(
                status_code=503,
                detail="Task upload is disabled until MARL_API_TOKEN is configured.",
            )
        supplied_token = request.headers.get("X-API-Token")
        if not supplied_token or not secrets.compare_digest(
            supplied_token, self.api_token
        ):
            raise HTTPException(status_code=401, detail="invalid or missing API token")
        if not TASK_SCRIPT_PATTERN.fullmatch(filename) or Path(filename).name != filename:
            raise HTTPException(
                status_code=422,
                detail="task uploads must be named Python scripts (safe basename ending in .py)",
            )
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="task script is too large")
            except ValueError as error:
                raise HTTPException(status_code=400, detail="invalid content-length") from error
        self.task_directory.mkdir(parents=True, exist_ok=True)
        destination = self.task_directory / filename
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".upload-", suffix=".tmp", dir=destination.parent
        )
        digest = hashlib.sha256()
        total_bytes = 0
        try:
            with os.fdopen(descriptor, "wb") as output:
                async for chunk in request.stream():
                    total_bytes += len(chunk)
                    if total_bytes > self.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="task script is too large")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if total_bytes == 0:
                raise HTTPException(status_code=422, detail="task script cannot be empty")
            hints = _upload_hints(request.headers)
            with self.submission_lock:
                task_id = Path(filename).stem
                hints_path = self.task_directory / f"{task_id}.hints.json"
                if destination.exists() or (self.task_directory / f"{task_id}.json").exists():
                    raise HTTPException(
                        status_code=409,
                        detail=f"task {task_id!r} already exists",
                    )
                os.replace(temporary_name, destination)
                try:
                    _write_ticket_atomically(hints_path, hints)
                except OSError:
                    destination.unlink(missing_ok=True)
                    raise
                stat = destination.stat()
                self.observed_uploads[filename] = (
                    stat.st_size,
                    stat.st_mtime_ns,
                    time.monotonic() - self.upload_stability_sec,
                )
        except Exception:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise
        self.scan_uploaded_scripts()
        ticket_path = self.task_directory / f"{Path(filename).stem}.json"
        scheduler_status = (
            "assigned" if ticket_path.is_file() else "queued_for_auto_assignment"
        )
        return {
            "status": "uploaded",
            "task_id": Path(filename).stem,
            "script_name": filename,
            "size_bytes": str(total_bytes),
            "sha256": digest.hexdigest(),
            "scheduler_status": scheduler_status,
        }

    def scan_uploaded_scripts(self) -> int:
        """Turn stable unassigned Python uploads into scheduler JSON tickets."""
        with self.submission_lock:
            return self._scan_uploaded_scripts_unlocked()

    def _scan_uploaded_scripts_unlocked(self) -> int:
        if not self.task_directory.is_dir():
            return 0
        now = time.monotonic()
        current_names = set()
        stable_uploads = []
        for script_path in self.task_directory.glob("*.py"):
            if not script_path.is_file() or script_path.resolve().parent != self.task_directory.resolve():
                continue
            if not TASK_SCRIPT_PATTERN.fullmatch(script_path.name):
                LOGGER.warning("Ignoring task script with invalid basename: %s", script_path.name)
                continue
            task_id = script_path.stem
            if (self.task_directory / f"{task_id}.json").exists():
                continue
            try:
                stat = script_path.stat()
            except OSError:
                LOGGER.warning("Could not inspect uploaded task script %s", script_path)
                continue
            current_names.add(script_path.name)
            previous = self.observed_uploads.get(script_path.name)
            signature = (stat.st_size, stat.st_mtime_ns)
            if previous is None or previous[:2] != signature:
                self.observed_uploads[script_path.name] = (*signature, now)
            elif now - previous[2] >= self.upload_stability_sec:
                stable_uploads.append((script_path.name, stat.st_size))
        self.observed_uploads = {
            filename: observation
            for filename, observation in self.observed_uploads.items()
            if filename in current_names
        }

        available_scripts = {
            path.name
            for path in self.task_directory.glob("*.py")
            if path.is_file()
            and path.resolve().parent == self.task_directory.resolve()
            and TASK_SCRIPT_PATTERN.fullmatch(path.name)
        }
        stable_names = {filename for filename, _ in stable_uploads}
        import_dependencies = {}
        analyzable_names = set()
        for filename in stable_names:
            try:
                import_dependencies.update(
                    detect_import_dependencies(
                        {filename: self.task_directory / filename},
                        available_scripts,
                    )
                )
                analyzable_names.add(filename)
            except DependencyAnalysisError:
                LOGGER.exception(
                    "Could not statically analyze imports in task %s; "
                    "scheduling without inferred dependencies",
                    filename,
                )
                import_dependencies[filename] = ()
                analyzable_names.add(filename)

        ready_uploads = []
        for filename, file_size in stable_uploads:
            if filename not in analyzable_names:
                continue
            unresolved_parents = [
                parent
                for parent in import_dependencies[filename]
                if parent not in stable_names
                and not (self.task_directory / f"{Path(parent).stem}.json").is_file()
            ]
            if unresolved_parents:
                LOGGER.info(
                    "Deferring task %s until imported task(s) are stable: %s",
                    filename,
                    ", ".join(unresolved_parents),
                )
                continue
            ready_uploads.append((filename, file_size))

        scheduled = 0
        pending_by_name = dict(ready_uploads)
        while pending_by_name:
            component_names = {min(pending_by_name)}
            changed = True
            while changed:
                changed = False
                for filename in tuple(pending_by_name):
                    linked = set(import_dependencies[filename])
                    linked.update(
                        child
                        for child, parents in import_dependencies.items()
                        if filename in parents
                    )
                    if component_names.intersection(linked) and filename not in component_names:
                        component_names.add(filename)
                        changed = True

            component_uploads = [
                (filename, pending_by_name.pop(filename))
                for filename in sorted(component_names)
                if filename in pending_by_name
            ]
            try:
                tasks = []
                for filename, file_size in component_uploads:
                    task_id = Path(filename).stem
                    hints_path = self.task_directory / f"{task_id}.hints.json"
                    if hints_path.exists():
                        hints = json.loads(hints_path.read_text(encoding="utf-8"))
                        if not isinstance(hints, dict):
                            raise ValueError("task hints must be a JSON object")
                    else:
                        hints = {}
                    parents = [
                        Path(parent).stem
                        for parent in import_dependencies[filename]
                    ]
                    tasks.append(
                        SubmittedTask(
                            task_id=task_id,
                            script_name=filename,
                            file_size=file_size / 1024,
                            disk_size=file_size / 1024,
                            doc_type=hints.get("doc_type", "text"),
                            parents=parents,
                            baseline_cpu_req=hints.get("baseline_cpu_req", 0.0),
                            baseline_gpu_req=hints.get("baseline_gpu_req", 0.0),
                        )
                    )
                existing_tasks = [
                    task
                    for task in tasks
                    if not (self.task_directory / f"{task.task_id}.json").exists()
                ]
                if not existing_tasks:
                    continue
                result = self._submit_unlocked(TaskBatch(tasks=existing_tasks))
                LOGGER.info(
                    "Assigned uploaded task group %s: %s",
                    ", ".join(task.task_id for task in existing_tasks),
                    result,
                )
                scheduled += len(existing_tasks)
                for task in existing_tasks:
                    self.observed_uploads.pop(task.script_name, None)
                    self.schedule_retry_log_times.pop(task.script_name, None)
            except HTTPException as error:
                for task in tasks:
                    if error.status_code in (422, 503):
                        now = time.monotonic()
                        last_log = self.schedule_retry_log_times.get(
                            task.script_name, 0.0
                        )
                        if now - last_log >= SCHEDULE_RETRY_LOG_INTERVAL_SEC:
                            LOGGER.warning(
                                "Uploaded task %s is waiting for an eligible scheduler: %s",
                                task.script_name,
                                error.detail,
                            )
                            self.schedule_retry_log_times[task.script_name] = now
                    else:
                        LOGGER.warning(
                            "Could not schedule uploaded task %s: %s",
                            task.script_name,
                            error.detail,
                        )
            except (OSError, ValueError, RuntimeError):
                LOGGER.exception(
                    "Unexpected failure scheduling uploaded task group %s",
                    ", ".join(task.script_name for task in tasks),
                )
        return scheduled

    def _submit_unlocked(self, batch: TaskBatch) -> Dict[str, Any]:
        if self.models is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Live scheduler models are not configured. Set "
                    "MARL_PREDICTOR_PATH, MARL_GNN_PATH, and MARL_SCHEDULER_PATH."
                ),
            )
        task_ids = [task.task_id for task in batch.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise HTTPException(status_code=422, detail="task IDs must be unique")
        task_id_set = set(task_ids)
        for task in batch.tasks:
            for parent in task.parents:
                if not TASK_SCRIPT_PATTERN.fullmatch(f"{parent}.py"):
                    raise HTTPException(
                        status_code=422,
                        detail=f"invalid dependency task ID: {parent!r}",
                    )
        try:
            edge_index = build_dependency_edges(
                task_ids,
                {
                    task.task_id: tuple(
                        parent for parent in task.parents if parent in task_id_set
                    )
                    for task in batch.tasks
                },
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        for task in batch.tasks:
            for parent in task.parents:
                if parent not in task_id_set and not (
                    self.task_directory / f"{parent}.json"
                ).is_file():
                    raise HTTPException(
                        status_code=422,
                        detail=f"dependency task {parent!r} has no task ticket",
                    )

        scripts = {}
        for task in batch.tasks:
            script_name = task.script_name
            if (
                Path(script_name).name != script_name
                or not TASK_SCRIPT_PATTERN.fullmatch(script_name)
            ):
                raise HTTPException(
                    status_code=422,
                    detail=f"script_name for task {task.task_id!r} must be a .py basename",
                )
            script_path = self.task_directory / script_name
            if (
                not script_path.is_file()
                or script_path.resolve().parent != self.task_directory.resolve()
            ):
                raise HTTPException(
                    status_code=404,
                    detail=f"script {script_name!r} was not found in the task share",
                )
            ticket_path = self.task_directory / f"{task.task_id}.json"
            if ticket_path.exists():
                raise HTTPException(
                    status_code=409,
                    detail=f"task ID {task.task_id!r} has already been submitted",
                )
            scripts[task.task_id] = script_name

        active_nodes = self._active_nodes()
        if not active_nodes:
            raise HTTPException(status_code=503, detail="no live scheduler nodes")
        infra = _infrastructure_context(active_nodes)
        task_inputs = [
            TaskInput(
                task_id=task.task_id,
                file_size=task.file_size,
                disk_size=task.disk_size,
                doc_type=task.doc_type,
                parents=tuple(task.parents),
                baseline_cpu_req=task.baseline_cpu_req,
                baseline_gpu_req=task.baseline_gpu_req,
            )
            for task in batch.tasks
        ]
        baseline_predictions = self.models.predictor.predict(
            task_inputs, [infra] * len(task_inputs)
        )
        features = task_gnn_feature_tensor(
            task_inputs, [infra] * len(task_inputs), baseline_predictions
        )
        device = next(self.models.encoder.parameters()).device
        refined_features, embeddings = self.models.encoder.predict_resources(
            features.to(device), edge_index.to(device)
        )
        embeddings = embeddings.to(device="cpu", dtype=torch.float32)
        predictions = [
            PredictedTask(
                task_id=prediction.task_id,
                parents=prediction.parents,
                target_exec_time=float(refined[0]),
                target_cpu_req=max(
                    prediction.target_cpu_req, float(refined[1])
                ),
                target_gpu_req=(
                    max(prediction.target_gpu_req, float(refined[2]))
                    if task.baseline_gpu_req > 0
                    else 0.0
                ),
                target_ram_req=float(refined[3]),
            )
            for task, prediction, refined in zip(
                task_inputs, baseline_predictions, refined_features.cpu()
            )
        ]
        allocation_order = _topological_order(task_ids, batch.tasks)
        prediction_by_id = {prediction.task_id: prediction for prediction in predictions}
        submitted_by_id = {task.task_id: task for task in batch.tasks}
        positions = {task_id: index for index, task_id in enumerate(task_ids)}
        remaining = {
            node_id: _node_capacity(node)
            for node_id, node in active_nodes.items()
        }
        reserved = self._pending_reservations(active_nodes)
        for node_id, reservation in reserved.items():
            for resource in ("cpu", "gpu", "ram"):
                remaining[node_id][resource] = max(
                    0.0, remaining[node_id][resource] - reservation[resource]
                )
        max_cost = max(
            (node["metrics"].get("cost_per_second", 0.03) for node in active_nodes.values()),
            default=0.03,
        )
        assignments = []
        training_data = {}
        queued_elsewhere = self._queued_task_count(set(task_ids))
        for task_id in allocation_order:
            prediction = prediction_by_id[task_id]
            candidates = tuple(active_nodes)
            telemetry_rows = []
            feasible_rows = []
            for node_id in candidates:
                node = active_nodes[node_id]
                capacity = remaining[node_id]
                telemetry_rows.append(
                    _node_telemetry(node, capacity, cost_scale=max_cost)
                )
                feasible_rows.append(_fits(prediction, capacity))
            mask = torch.tensor([feasible_rows], dtype=torch.bool)
            if not mask.any():
                raise HTTPException(
                    status_code=422,
                    detail=f"no live node has enough capacity for task {task_id!r}",
                )
            state = build_joint_state(
                embeddings[positions[task_id]].unsqueeze(0),
                torch.tensor(telemetry_rows, dtype=torch.float32),
                task_ids=(task_id,),
                server_ids=candidates,
                feasibility_mask=mask,
                device=self.models.scheduler.device,
            )
            with self.model_lock:
                decision = self.models.scheduler.select_actions(state)
            node_id = candidates[decision.action_indices[0]]
            chosen_node = active_nodes[node_id]
            metrics = chosen_node["metrics"]
            load_before = max(
                metrics["cpu_percent"] / 100,
                metrics["ram_percent"] / 100,
                metrics.get("gpu_percent", 0) / 100,
            )
            training_data[task_id] = (
                state,
                decision.action_indices,
                load_before,
                queued_elsewhere + len(allocation_order) - len(assignments) - 1,
                metrics.get("cost_per_second", 0.03),
            )
            _reserve(prediction, remaining[node_id])
            assignments.append(
                {
                    "task_id": task_id,
                    "node_id": node_id,
                    "script_name": scripts[task_id],
                    "prediction": {
                        "target_exec_time": prediction.target_exec_time,
                        "target_cpu_req": prediction.target_cpu_req,
                        "target_gpu_req": prediction.target_gpu_req,
                        "target_ram_req": prediction.target_ram_req,
                    },
                    "gnn_embedding": embeddings[positions[task_id]].tolist(),
                    "feasible_nodes": {
                        candidate: bool(feasible_rows[index])
                        for index, candidate in enumerate(candidates)
                    },
                    "scheduler_value": decision.value,
                    "ticket": f"{task_id}.json",
                }
            )

        tickets = []
        self.task_directory.mkdir(parents=True, exist_ok=True)
        for assignment in assignments:
            task = submitted_by_id[assignment["task_id"]]
            ticket = {
                **task.model_dump(),
                "assigned_to": assignment["node_id"],
                "status": "PENDING",
                "created_at": time.time(),
                "prediction": assignment["prediction"],
                "gnn_embedding": assignment["gnn_embedding"],
                "feasible_nodes": assignment["feasible_nodes"],
                "scheduler_value": assignment["scheduler_value"],
                "depends_on": task.parents,
            }
            ticket_path = self.task_directory / assignment["ticket"]
            _write_ticket_atomically(ticket_path, ticket)
            tickets.append(assignment)
        with self.experience_lock:
            self.pending_experiences.update(training_data)
        return {"status": "accepted", "assignments": tickets}

    def _active_nodes(self) -> Dict[str, Dict[str, Any]]:
        self.refresh_status()
        now = time.time()
        return {
            node_id: node
            for node_id, node in self.cluster_state.items()
            if node["status"] in ("FREE", "OCCUPIED", "HIGH_LOAD")
            and now - node["last_seen"] <= self.heartbeat_timeout_sec
        }

    def _pending_reservations(
        self, active_nodes: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        reservations = {
            node_id: {"cpu": 0.0, "gpu": 0.0, "ram": 0.0}
            for node_id in active_nodes
        }
        if not self.task_directory.exists():
            return reservations
        for ticket_path in self.task_directory.glob("*.json"):
            try:
                ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise HTTPException(
                    status_code=500,
                    detail=f"cannot inspect pending ticket {ticket_path.name}: {error}",
                ) from error
            node_id = ticket.get("assigned_to")
            if ticket.get("status") not in ("PENDING", "RUNNING") or node_id not in reservations:
                continue
            prediction = ticket.get("prediction")
            if not isinstance(prediction, dict):
                raise HTTPException(
                    status_code=500,
                    detail=f"pending ticket {ticket_path.name} has no resource prediction",
                )
            for resource, key in (
                ("cpu", "target_cpu_req"),
                ("gpu", "target_gpu_req"),
                ("ram", "target_ram_req"),
            ):
                value = prediction.get(key)
                if not isinstance(value, (int, float)) or value < 0:
                    raise HTTPException(
                        status_code=500,
                        detail=f"pending ticket {ticket_path.name} has invalid {key}",
                    )
                reservations[node_id][resource] += float(value)
        return reservations

    def _queued_task_count(self, excluded_task_ids: set) -> int:
        if not self.task_directory.is_dir():
            return 0
        count = 0
        for script_path in self.task_directory.glob("*.py"):
            if (
                TASK_SCRIPT_PATTERN.fullmatch(script_path.name)
                and script_path.stem not in excluded_task_ids
                and not (self.task_directory / f"{script_path.stem}.json").exists()
            ):
                count += 1
        for ticket_path in self.task_directory.glob("*.json"):
            try:
                ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise HTTPException(
                    status_code=500,
                    detail=f"cannot inspect queued ticket {ticket_path.name}: {error}",
                ) from error
            if (
                ticket_path.stem not in excluded_task_ids
                and isinstance(ticket, dict)
                and ticket.get("status") in ("PENDING", "RUNNING")
            ):
                count += 1
        return count

    def collect_finished_tasks(self) -> None:
        """Batch terminal execution feedback into one PPO policy update."""
        if self.models is None:
            return
        with self.experience_lock:
            pending = list(self.pending_experiences.items())
        ready = []
        for task_id, training in pending:
            ticket_path = self.task_directory / f"{task_id}.json"
            try:
                ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                LOGGER.warning("Cannot read task ticket %s: %s", task_id, error)
                continue
            status = ticket.get("status")
            if status not in ("COMPLETED", "FAILED"):
                continue
            state, actions, load_before, tasks_waiting, cost_rate = training
            if status == "COMPLETED":
                elapsed = ticket.get("execution_time_sec")
                if not isinstance(elapsed, (int, float)) or elapsed < 0:
                    LOGGER.error("Task %s has invalid execution_time_sec", task_id)
                    continue
                adaptive = self.models.scheduler.compute_reward(
                    actual_execution_time=float(elapsed),
                    actual_cost=float(elapsed) * cost_rate,
                    system_load=min(max(load_before, 0.0), 1.0),
                    tasks_waiting=tasks_waiting,
                )
                reward = adaptive.reward
            else:
                reward = -1.0
            ready.append(
                (
                    task_id,
                    ticket_path,
                    ticket,
                    SchedulerExperience(
                        state=state,
                        action_indices=actions,
                        reward=reward,
                        done=True,
                    )
                )
            )
        if not ready:
            return

        try:
            with self.model_lock:
                update_metrics = self.models.scheduler.update(
                    [item[3] for item in ready]
                )
        except (ValueError, RuntimeError):
            LOGGER.exception("Batched online scheduler update failed")
            return

        checkpoint_saved = False
        checkpoint_error = None
        if self.scheduler_checkpoint_path:
            try:
                with self.model_lock:
                    self.models.scheduler.save(self.scheduler_checkpoint_path)
                checkpoint_saved = True
            except (OSError, RuntimeError) as error:
                LOGGER.exception("Could not persist scheduler checkpoint")
                checkpoint_error = str(error)

        for task_id, ticket_path, ticket, experience in ready:
            ticket["online_reward"] = experience.reward
            ticket["scheduler_feedback_recorded"] = True
            ticket["scheduler_update_batch_size"] = update_metrics["transitions"]
            if self.scheduler_checkpoint_path:
                ticket["scheduler_checkpoint_saved"] = checkpoint_saved
                if checkpoint_error:
                    ticket["scheduler_checkpoint_error"] = checkpoint_error
            _write_ticket_atomically(ticket_path, ticket)
            with self.experience_lock:
                self.pending_experiences.pop(task_id, None)


def _infrastructure_context(
    nodes: Dict[str, Dict[str, Any]]
) -> InfrastructureContext:
    metrics = [node["metrics"] for node in nodes.values()]
    bandwidth = sum(
        max(item.get("net_tx_kbps", 0), item.get("net_rx_kbps", 0), 1)
        for item in metrics
    ) / len(metrics)
    latency = sum(item.get("rtt_ms", 0) for item in metrics) / len(metrics)
    available_ram = sum(item.get("ram_free_gb", 0) for item in metrics) / len(metrics)
    return InfrastructureContext(
        available_bandwidth=bandwidth,
        latency=latency,
        available_ram=available_ram,
    )


def _upload_hints(headers: Any) -> Dict[str, Any]:
    doc_type = headers.get("X-Doc-Type", "text").strip().lower()
    if doc_type not in ("text", "image", "video", "archive"):
        raise HTTPException(
            status_code=422,
            detail="X-Doc-Type must be text, image, video, or archive",
        )
    hints: Dict[str, Any] = {"doc_type": doc_type}
    for header, key in (
        ("X-Baseline-CPU-Req", "baseline_cpu_req"),
        ("X-Baseline-GPU-Req", "baseline_gpu_req"),
    ):
        raw = headers.get(header, "0")
        try:
            value = float(raw)
        except ValueError as error:
            raise HTTPException(
                status_code=422, detail=f"{header} must be a non-negative number"
            ) from error
        if not isfinite(value) or value < 0:
            raise HTTPException(
                status_code=422, detail=f"{header} must be a non-negative number"
            )
        hints[key] = value
    return hints


def _node_capacity(node: Dict[str, Any]) -> Dict[str, float]:
    metrics = node["metrics"]
    return {
        "cpu": max(metrics["cores_logical"] * (1 - metrics["cpu_percent"] / 100), 0),
        "gpu": metrics.get("gpu_count", 0)
        * (1 - metrics.get("gpu_percent", 0) / 100),
        "ram": metrics["ram_free_gb"],
    }


def _node_telemetry(
    node: Dict[str, Any],
    remaining: Dict[str, float],
    *,
    cost_scale: float = 1.0,
) -> List[float]:
    metrics = node["metrics"]
    cpu_total = max(float(metrics["cores_logical"]), 1.0)
    ram_total = max(float(metrics["ram_total_gb"]), 0.1)
    gpu_total = max(float(metrics.get("gpu_count", 0)), 1.0)
    bandwidth = max(
        metrics.get("net_tx_kbps", 0), metrics.get("net_rx_kbps", 0)
    )
    return [
        remaining["cpu"] / cpu_total,
        remaining["gpu"] / gpu_total,
        remaining["ram"] / ram_total,
        bandwidth / 1000,
        metrics.get("rtt_ms", 0) / 120,
        max(metrics.get("cpu_percent", 0), metrics.get("ram_percent", 0)) / 100,
        metrics.get("cost_per_second", 0.03) / max(cost_scale, 1e-8),
    ]


def _fits(prediction: PredictedTask, capacity: Dict[str, float]) -> bool:
    return (
        prediction.target_cpu_req <= capacity["cpu"]
        and prediction.target_gpu_req <= capacity["gpu"]
        and prediction.target_ram_req <= capacity["ram"]
    )


def _reserve(prediction: PredictedTask, capacity: Dict[str, float]) -> None:
    capacity["cpu"] -= prediction.target_cpu_req
    capacity["gpu"] -= prediction.target_gpu_req
    capacity["ram"] -= prediction.target_ram_req


def _topological_order(
    task_ids: Sequence[str], tasks: Sequence[SubmittedTask]
) -> List[str]:
    task_id_set = set(task_ids)
    parents = {
        task.task_id: tuple(parent for parent in task.parents if parent in task_id_set)
        for task in tasks
    }
    remaining = set(task_ids)
    emitted = set()
    order = []
    while remaining:
        ready = [
            task_id
            for task_id in task_ids
            if task_id in remaining
            and all(parent in emitted for parent in parents[task_id])
        ]
        if not ready:
            raise HTTPException(status_code=422, detail="task dependencies contain a cycle")
        for task_id in ready:
            remaining.remove(task_id)
            emitted.add(task_id)
            order.append(task_id)
    return order


def _write_ticket_atomically(path: Path, ticket: Dict[str, Any]) -> None:
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(ticket, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        _replace_ticket_file(temporary_path, path)
    except Exception:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _replace_ticket_file(temporary_path: str, path: Path) -> None:
    for attempt in range(6):
        try:
            os.replace(temporary_path, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.02 * (attempt + 1))


def create_app(runtime: Optional[HubRuntime] = None) -> FastAPI:
    runtime = runtime or HubRuntime(models=load_model_bundle_from_environment())
    app = FastAPI(title="Cloud Resource Allocation Hub")
    app.state.runtime = runtime
    allowed_origins = [
        origin.strip()
        for origin in os.environ.get("MARL_DASHBOARD_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Token"],
        )

    @app.post("/api/telemetry")
    async def ingest_telemetry(packet: TelemetryPacket) -> Dict[str, str]:
        runtime.ingest(packet)
        return {"status": "ack"}

    @app.get("/")
    async def dashboard() -> FileResponse:
        return FileResponse(Path(__file__).with_name("dashboard.html"))

    @app.post("/api/tasks/upload")
    async def upload_task(
        request: Request,
        filename: str = Header(alias="X-File-Name"),
    ) -> Dict[str, str]:
        return await runtime.save_upload(request, filename)

    @app.get("/api/cluster/status")
    async def get_cluster_status() -> Dict[str, Any]:
        runtime.refresh_status()
        return runtime.cluster_state

    @app.get("/api/network/graph")
    async def get_network_graph() -> Dict[str, Any]:
        return runtime.network_graph()

    @app.post("/api/tasks/submit")
    async def submit_tasks(
        batch: TaskBatch,
        api_token: Optional[str] = Header(default=None, alias="X-API-Token"),
    ) -> Dict[str, Any]:
        return runtime.submit(batch, api_token)

    @app.get("/api/tasks/{task_id}")
    async def get_task_status(task_id: str) -> Dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", task_id):
            raise HTTPException(status_code=422, detail="invalid task_id")
        ticket_path = runtime.task_directory / f"{task_id}.json"
        if not ticket_path.is_file():
            raise HTTPException(status_code=404, detail="task not found")
        try:
            return json.loads(ticket_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise HTTPException(status_code=500, detail="task ticket cannot be read") from error

    @app.on_event("startup")
    async def start_feedback_monitor() -> None:
        last_feedback_collection = 0.0

        async def monitor() -> None:
            nonlocal last_feedback_collection
            while True:
                try:
                    await asyncio.to_thread(runtime.scan_uploaded_scripts)
                    now = time.monotonic()
                    if now - last_feedback_collection >= 1.0:
                        await asyncio.to_thread(runtime.collect_finished_tasks)
                        last_feedback_collection = now
                except (HTTPException, OSError, RuntimeError, ValueError):
                    LOGGER.exception("Hub task scan or feedback cycle failed")
                await asyncio.sleep(UPLOAD_SCAN_INTERVAL_SEC)

        app.state.feedback_task = asyncio.create_task(monitor())

    @app.on_event("shutdown")
    async def stop_feedback_monitor() -> None:
        task = getattr(app.state, "feedback_task", None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "hub:app",
        host=os.environ.get("MARL_HUB_HOST", "0.0.0.0"),
        port=int(os.environ.get("MARL_HUB_PORT", "8000")),
        reload=False,
    )
