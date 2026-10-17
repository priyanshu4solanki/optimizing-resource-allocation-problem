"""Local four-agent upload-to-result integration test."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from agent import active_task_count
from gnn_encoder import TaskGraphEncoder
from hub import (
    HubRuntime,
    Metrics,
    ModelBundle,
    TelemetryPacket,
    create_app,
)
from marl_scheduler import MARLScheduler
from task_predictors import TaskResourcePredictor, generate_synthetic_training_data
from task_runner import _dependencies, run_worker


class LiveTaskFlowTest(unittest.TestCase):
    def test_folder_scan_detects_import_dependencies_and_worker_waits(self) -> None:
        training_tasks, training_contexts = generate_synthetic_training_data(
            sample_count=32, seed=103
        )
        predictor = TaskResourcePredictor(random_state=103)
        predictor.fit_synthetic(training_tasks, training_contexts)
        models = ModelBundle(
            predictor=predictor,
            encoder=TaskGraphEncoder(input_dim=12, hidden_dim=8, embedding_dim=8),
            scheduler=MARLScheduler(
                input_dim=15, hidden_dim=16, update_epochs=1
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            task_directory = Path(temporary_directory) / "cluster_task"
            task_directory.mkdir()
            (task_directory / "provider.py").write_text(
                "def value():\n    return 42\n", encoding="utf-8"
            )
            (task_directory / "consumer.py").write_text(
                "from provider import value\nprint(value())\n",
                encoding="utf-8",
            )
            runtime = HubRuntime(
                node_ids=("CPU1",),
                models=models,
                task_directory=task_directory,
                heartbeat_timeout_sec=60,
                upload_stability_sec=0,
            )
            runtime.ingest(
                TelemetryPacket(
                    node_id="CPU1",
                    ip_address="127.0.0.1",
                    timestamp=time.time(),
                    status="FREE",
                    metrics=Metrics(
                        cpu_percent=5,
                        ram_percent=20,
                        ram_free_gb=8,
                        ram_total_gb=16,
                        active_tasks=0,
                        net_tx_kbps=10,
                        net_rx_kbps=10,
                        rtt_ms=1,
                        cores_logical=8,
                        gpu_count=0,
                        gpu_percent=0,
                        cost_per_second=0.03,
                    ),
                )
            )

            self.assertEqual(runtime.scan_uploaded_scripts(), 0)
            self.assertEqual(runtime.scan_uploaded_scripts(), 2)
            provider_ticket = json.loads(
                (task_directory / "provider.json").read_text(encoding="utf-8")
            )
            consumer_ticket = json.loads(
                (task_directory / "consumer.json").read_text(encoding="utf-8")
            )
            self.assertEqual(consumer_ticket["depends_on"], ["provider"])
            self.assertEqual(_dependencies(consumer_ticket, task_directory), "waiting")

            provider_ticket["status"] = "COMPLETED"
            (task_directory / "provider.json").write_text(
                json.dumps(provider_ticket), encoding="utf-8"
            )
            self.assertEqual(_dependencies(consumer_ticket, task_directory), "ready")

    def test_folder_scan_schedules_script_with_syntax_error_without_dependencies(self) -> None:
        training_tasks, training_contexts = generate_synthetic_training_data(
            sample_count=32, seed=104
        )
        predictor = TaskResourcePredictor(random_state=104)
        predictor.fit_synthetic(training_tasks, training_contexts)
        models = ModelBundle(
            predictor=predictor,
            encoder=TaskGraphEncoder(input_dim=12, hidden_dim=8, embedding_dim=8),
            scheduler=MARLScheduler(
                input_dim=15, hidden_dim=16, update_epochs=1
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            task_directory = Path(temporary_directory) / "cluster_task"
            task_directory.mkdir()
            (task_directory / "broken.py").write_text("if:\n", encoding="utf-8")
            runtime = HubRuntime(
                node_ids=("CPU1",),
                models=models,
                task_directory=task_directory,
                heartbeat_timeout_sec=60,
                upload_stability_sec=0,
            )
            runtime.ingest(
                TelemetryPacket(
                    node_id="CPU1",
                    ip_address="127.0.0.1",
                    timestamp=time.time(),
                    status="FREE",
                    metrics=Metrics(
                        cpu_percent=5,
                        ram_percent=20,
                        ram_free_gb=8,
                        ram_total_gb=16,
                        active_tasks=0,
                        net_tx_kbps=10,
                        net_rx_kbps=10,
                        rtt_ms=1,
                        cores_logical=8,
                        gpu_count=0,
                        gpu_percent=0,
                        cost_per_second=0.03,
                    ),
                )
            )

            self.assertEqual(runtime.scan_uploaded_scripts(), 0)
            self.assertEqual(runtime.scan_uploaded_scripts(), 1)
            ticket = json.loads(
                (task_directory / "broken.json").read_text(encoding="utf-8")
            )
            self.assertEqual(ticket["depends_on"], [])

    def test_unhinted_python_task_is_schedulable_on_cpu_only_nodes(self) -> None:
        training_tasks, training_contexts = generate_synthetic_training_data(
            sample_count=32, seed=102
        )
        predictor = TaskResourcePredictor(random_state=102)
        predictor.fit_synthetic(training_tasks, training_contexts)
        models = ModelBundle(
            predictor=predictor,
            encoder=TaskGraphEncoder(input_dim=12, hidden_dim=8, embedding_dim=8),
            scheduler=MARLScheduler(
                input_dim=15, hidden_dim=16, update_epochs=1
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            task_directory = Path(temporary_directory) / "cluster_task"
            task_directory.mkdir()
            (task_directory / "cpu_job.py").write_text(
                "print('CPU task completed')\n", encoding="utf-8"
            )
            runtime = HubRuntime(
                node_ids=("CPU1",),
                models=models,
                task_directory=task_directory,
                heartbeat_timeout_sec=60,
                upload_stability_sec=0,
            )
            runtime.ingest(
                TelemetryPacket(
                    node_id="CPU1",
                    ip_address="127.0.0.1",
                    timestamp=time.time(),
                    status="FREE",
                    metrics=Metrics(
                        cpu_percent=5,
                        ram_percent=20,
                        ram_free_gb=8,
                        ram_total_gb=16,
                        active_tasks=0,
                        net_tx_kbps=10,
                        net_rx_kbps=10,
                        rtt_ms=1,
                        cores_logical=8,
                        gpu_count=0,
                        gpu_percent=0,
                        cost_per_second=0.03,
                    ),
                )
            )

            self.assertEqual(runtime.scan_uploaded_scripts(), 0)
            self.assertEqual(runtime.scan_uploaded_scripts(), 1)
            ticket = json.loads(
                (task_directory / "cpu_job.json").read_text(encoding="utf-8")
            )
            self.assertEqual(ticket["assigned_to"], "CPU1")
            self.assertEqual(ticket["status"], "PENDING")
            self.assertEqual(ticket["prediction"]["target_gpu_req"], 0)

    def test_four_agents_gpu_hint_task_runner_and_result_feedback(self) -> None:
        training_tasks, training_contexts = generate_synthetic_training_data(
            sample_count=32, seed=101
        )
        predictor = TaskResourcePredictor(random_state=101)
        predictor.fit_synthetic(training_tasks, training_contexts)
        models = ModelBundle(
            predictor=predictor,
            encoder=TaskGraphEncoder(input_dim=12, hidden_dim=8, embedding_dim=8),
            scheduler=MARLScheduler(
                input_dim=15, hidden_dim=16, update_epochs=1
            ),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            task_directory = root / "cluster_task"
            result_directory = root / "cluster_result"
            task_directory.mkdir()
            runtime = HubRuntime(
                node_ids=("S1", "S2", "S3", "S4"),
                models=models,
                task_directory=task_directory,
                heartbeat_timeout_sec=60,
                api_token="integration-test-token",
                upload_stability_sec=0,
            )
            self._send_four_agent_heartbeats(runtime)
            stop_event = threading.Event()
            workers = [
                threading.Thread(
                    target=run_worker,
                    args=(node_id, task_directory, result_directory),
                    kwargs={
                        "poll_interval": 0.02,
                        "timeout_seconds": 10,
                        "stop_event": stop_event,
                    },
                    daemon=True,
                )
                for node_id in ("S1", "S2", "S3", "S4")
            ]
            self.addCleanup(lambda: [worker.join(timeout=2) for worker in workers])
            self.addCleanup(stop_event.set)

            with TestClient(create_app(runtime)) as client:
                self.assertEqual(client.get("/").status_code, 200)
                response = client.post(
                    "/api/tasks/upload",
                    content=self._task_script(root).encode("utf-8"),
                    headers={
                        "X-API-Token": "integration-test-token",
                        "X-File-Name": "gpu_job.py",
                        "X-Baseline-CPU-Req": "0.25",
                        "X-Baseline-GPU-Req": "0.5",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["scheduler_status"], "assigned")
                self.assertTrue((task_directory / "gpu_job.json").is_file())
                ticket = json.loads(
                    (task_directory / "gpu_job.json").read_text(encoding="utf-8")
                )
                self.assertIn(ticket["assigned_to"], ("S3", "S4"))
                self.assertGreaterEqual(ticket["prediction"]["target_gpu_req"], 0.5)
                self.assertEqual(ticket["baseline_gpu_req"], 0.5)
                self.assertEqual(ticket["baseline_cpu_req"], 0.25)
                self.assertEqual(len(ticket["gnn_embedding"]), 8)
                self.assertFalse(ticket["feasible_nodes"]["S1"])
                self.assertFalse(ticket["feasible_nodes"]["S2"])

                for worker in workers:
                    worker.start()
                self._wait_for_status(task_directory / "gpu_job.json")

                completed = json.loads(
                    (task_directory / "gpu_job.json").read_text(encoding="utf-8")
                )
                self.assertEqual(completed["status"], "COMPLETED")
                self.assertTrue(completed["scheduler_feedback_recorded"])
                self.assertTrue(
                    (result_directory / "gpu_job_result.txt").is_file()
                )
                self.assertEqual(
                    (root / "execution-count.txt").read_text(encoding="utf-8"),
                    "x",
                    "only the assigned worker should execute a ticket",
                )
                self.assertEqual(client.get("/api/tasks/gpu_job").status_code, 200)

            stop_event.set()
            for worker in workers:
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive(), "worker test thread did not stop")

            (task_directory / "folder_gpu.hints.json").write_text(
                json.dumps(
                    {
                        "doc_type": "text",
                        "baseline_cpu_req": 0.1,
                        "baseline_gpu_req": 0.25,
                    }
                ),
                encoding="utf-8",
            )
            (task_directory / "folder_gpu.py").write_text(
                "print('copied task')\n", encoding="utf-8"
            )
            self.assertEqual(runtime.scan_uploaded_scripts(), 0)
            self.assertEqual(runtime.scan_uploaded_scripts(), 1)
            folder_ticket = json.loads(
                (task_directory / "folder_gpu.json").read_text(encoding="utf-8")
            )
            self.assertIn(folder_ticket["assigned_to"], ("S3", "S4"))
            self.assertEqual(folder_ticket["baseline_gpu_req"], 0.25)
            self.assertTrue(folder_ticket["prediction"]["target_gpu_req"] >= 0.25)

    @staticmethod
    def _task_script(root: Path) -> str:
        count_file = str(root / "execution-count.txt")
        return (
            "from pathlib import Path\n"
            f"counter = Path({count_file!r})\n"
            "counter.write_text((counter.read_text() if counter.exists() else '') + 'x')\n"
            "print('normal Python task completed')\n"
        )

    @staticmethod
    def _send_four_agent_heartbeats(runtime: HubRuntime) -> None:
        reports = (
            ("S1", 0, 0.0, 20, 8, 16, 16, 15, 0.08),
            ("S2", 0, 0.0, 35, 12, 32, 12, 25, 0.05),
            ("S3", 1, 8.0, 25, 16, 32, 10, 18, 0.04),
            ("S4", 1, 4.0, 15, 24, 48, 8, 35, 0.10),
        )

        def report(row: tuple) -> None:
            (
                node_id,
                gpu_count,
                gpu_percent,
                ram_percent,
                ram_free,
                ram_total,
                cores,
                rtt,
                cost,
            ) = row
            runtime.ingest(
                TelemetryPacket(
                    node_id=node_id,
                    ip_address=f"10.0.0.{int(node_id[1:]) + 10}",
                    timestamp=time.time(),
                    status="FREE",
                    metrics=Metrics(
                        cpu_percent=8,
                        ram_percent=ram_percent,
                        ram_free_gb=ram_free,
                        ram_total_gb=ram_total,
                        active_tasks=active_task_count(runtime.task_directory),
                        net_tx_kbps=100 + cores,
                        net_rx_kbps=200 + cores,
                        rtt_ms=rtt,
                        cores_logical=cores,
                        gpu_count=gpu_count,
                        gpu_percent=gpu_percent,
                        cost_per_second=cost,
                    ),
                )
            )

        agents = [
            threading.Thread(target=report, args=(row,)) for row in reports
        ]
        for agent in agents:
            agent.start()
        for agent in agents:
            agent.join()
        if len(runtime._active_nodes()) != 4:
            raise AssertionError("all four simulated agents should be live")

    @staticmethod
    def _wait_for(path: Path, timeout: float = 10) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.is_file():
                return
            time.sleep(0.02)
        raise AssertionError(f"timed out waiting for {path.name}")

    @classmethod
    def _wait_for_status(
        cls, ticket_path: Path, timeout: float = 15
    ) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
                if ticket.get("scheduler_feedback_recorded"):
                    return
            except (OSError, json.JSONDecodeError):
                pass
            time.sleep(0.02)
        raise AssertionError("task did not complete and provide scheduler feedback")


if __name__ == "__main__":
    unittest.main()
