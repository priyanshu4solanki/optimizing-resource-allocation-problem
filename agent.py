"""Send this machine's resource telemetry to the allocation hub."""

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit

import psutil
import requests

LOGGER = logging.getLogger(__name__)


def local_ip() -> str:
    connection = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        connection.connect(("8.8.8.8", 80))
        return connection.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        connection.close()


def gpu_metrics() -> tuple[int, float]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return 0, 0.0
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return 0, 0.0
    try:
        values = [
            float(line.strip())
            for line in result.stdout.splitlines()
            if line.strip().replace(".", "", 1).isdigit()
        ]
    except ValueError:
        LOGGER.warning("Could not parse NVIDIA GPU utilization")
        return 0, 0.0
    return len(values), sum(values) / len(values) if values else 0.0


def active_task_count(task_directory: Path) -> int:
    """Count running task tickets visible on the worker's shared folder."""
    if not task_directory.is_dir():
        return 0
    running = 0
    for ticket_path in task_directory.glob("*.json"):
        if ticket_path.name.endswith(".hints.json"):
            continue
        try:
            ticket = json.loads(ticket_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning("Could not read task ticket %s: %s", ticket_path, error)
            continue
        if isinstance(ticket, dict) and ticket.get("status") == "RUNNING":
            running += 1
    return running


def run_agent(
    node_id: str, hub_url: str, interval: float, task_directory: Path
) -> None:
    parsed_url = urlsplit(hub_url)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.hostname:
        raise ValueError("hub URL must be a valid http(s) URL")
    host = parsed_url.hostname
    port = parsed_url.port or (443 if parsed_url.scheme == "https" else 80)
    ip_address = local_ip()
    print(f"[*] Agent {node_id} reporting to {hub_url}")

    previous_network = psutil.net_io_counters()
    previous_time = time.monotonic()
    psutil.cpu_percent(interval=None)
    time.sleep(1)
    while True:
        now = time.time()
        monotonic_now = time.monotonic()
        elapsed = max(monotonic_now - previous_time, 1e-6)
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        network = psutil.net_io_counters()
        tx_kbps = max(0.0, (network.bytes_sent - previous_network.bytes_sent) / 1024 / elapsed)
        rx_kbps = max(0.0, (network.bytes_recv - previous_network.bytes_recv) / 1024 / elapsed)
        previous_network, previous_time = network, monotonic_now
        gpu_count, gpu_percent = gpu_metrics()
        status = (
            "HIGH_LOAD"
            if cpu_percent > 85
            else "OCCUPIED"
            if cpu_percent > 25
            else "FREE"
        )
        payload = {
            "node_id": node_id,
            "ip_address": ip_address,
            "timestamp": now,
            "status": status,
            "metrics": {
                "cpu_percent": cpu_percent,
                "ram_percent": memory.percent,
                "ram_free_gb": round(memory.available / (1024**3), 3),
                "ram_total_gb": round(memory.total / (1024**3), 3),
                "active_tasks": active_task_count(task_directory),
                "net_tx_kbps": round(tx_kbps, 2),
                "net_rx_kbps": round(rx_kbps, 2),
                "rtt_ms": _measure_rtt(host, port),
                "cores_logical": psutil.cpu_count(logical=True) or 1,
                "gpu_count": gpu_count,
                "gpu_percent": gpu_percent,
                "cost_per_second": float(
                    os.environ.get("MARL_NODE_COST_PER_SECOND", "0.03")
                ),
            },
        }
        try:
            response = requests.post(hub_url, json=payload, timeout=2)
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"[!] Telemetry post failed: {error}")
        time.sleep(interval)


def _measure_rtt(host: str, port: int) -> float:
    connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connection.settimeout(0.5)
    started = time.perf_counter()
    try:
        connection.connect((host, port))
        return round((time.perf_counter() - started) * 1000, 2)
    except OSError:
        return 0.0
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", default=os.environ.get("MARL_NODE_ID", "S1"))
    parser.add_argument(
        "--hub-url",
        default=os.environ.get(
            "MARL_TELEMETRY_URL", "http://192.168.137.1:8000/api/telemetry"
        ),
    )
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument(
        "--task-directory",
        type=Path,
        default=Path(
            os.environ.get("MARL_TASKS_DIR", r"\\192.168.137.1\cluster_task")
        ),
    )
    arguments = parser.parse_args()
    if arguments.interval <= 0:
        parser.error("--interval must be positive")
    run_agent(
        arguments.node_id,
        arguments.hub_url,
        arguments.interval,
        arguments.task_directory,
    )


if __name__ == "__main__":
    main()
