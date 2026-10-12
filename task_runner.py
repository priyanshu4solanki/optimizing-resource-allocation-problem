"""Run hub-assigned Python task tickets from a shared task directory."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


def _write_ticket(ticket_path: Path, ticket: Dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{ticket_path.stem}.", suffix=".tmp", dir=ticket_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(ticket, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
        _replace_ticket_file(temporary_name, ticket_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _replace_ticket_file(temporary_name: str, ticket_path: Path) -> None:
    for attempt in range(6):
        try:
            os.replace(temporary_name, ticket_path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.02 * (attempt + 1))


def _load_ticket(ticket_path: Path) -> Dict[str, Any]:
    with ticket_path.open("r", encoding="utf-8") as ticket_file:
        ticket = json.load(ticket_file)
    if not isinstance(ticket, dict):
        raise ValueError("ticket must contain a JSON object")
    return ticket


def _dependencies(ticket: Dict[str, Any], task_directory: Path) -> str:
    for task_id in ticket.get("depends_on", []):
        if not isinstance(task_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", task_id
        ):
            raise ValueError(f"invalid dependency task ID: {task_id!r}")
        parent_path = task_directory / f"{task_id}.json"
        if not parent_path.is_file():
            return "waiting"
        parent = _load_ticket(parent_path)
        if parent.get("status") == "FAILED":
            return "failed"
        if parent.get("status") != "COMPLETED":
            return "waiting"
    return "ready"


def _claim_ticket(ticket_path: Path, timeout_seconds: float) -> Optional[Path]:
    claim_path = ticket_path.with_suffix(".claim")
    stale_after = max(60.0, timeout_seconds * 2)
    for attempt in range(2):
        try:
            descriptor = os.open(
                claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            try:
                if time.time() - claim_path.stat().st_mtime <= stale_after:
                    return None
                claim_path.unlink()
            except FileNotFoundError:
                continue
            if attempt == 1:
                return None
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as claim:
                claim.write(f"{os.getpid()} {time.time()}\n")
            return claim_path
    return None


def _run_ticket(
    ticket_path: Path,
    ticket: Dict[str, Any],
    node_id: str,
    timeout_seconds: float,
    results_directory: Path,
) -> None:
    script_name = ticket.get("script_name")
    if (
        not isinstance(script_name, str)
        or Path(script_name).name != script_name
        or not re.fullmatch(r"[A-Za-z0-9_.-]+\.py", script_name)
    ):
        raise ValueError("ticket script_name must be a .py basename")
    task_directory = ticket_path.parent.resolve()
    script_path = (task_directory / script_name).resolve()
    if script_path.parent != task_directory or not script_path.is_file():
        raise ValueError("script must exist directly inside the task directory")
    results_directory.mkdir(parents=True, exist_ok=True)

    ticket["status"] = "RUNNING"
    ticket["started_at"] = time.time()
    _write_ticket(ticket_path, ticket)
    started = time.monotonic()
    try:
        process = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(task_directory),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = round(time.monotonic() - started, 3)
        result_name = f"{ticket_path.stem}_result.txt"
        result_path = results_directory / result_name
        with result_path.open("w", encoding="utf-8") as output:
            output.write(f"Node: {node_id}\nRuntime: {elapsed}s\n")
            output.write(f"Exit Code: {process.returncode}\n\n{process.stdout}")
            if process.stderr:
                output.write(f"\n[stderr]\n{process.stderr}")
        ticket.update(
            {
                "status": "COMPLETED" if process.returncode == 0 else "FAILED",
                "execution_time_sec": elapsed,
                "result_file": result_name,
                "exit_code": process.returncode,
                "finished_at": time.time(),
            }
        )
        if process.returncode:
            ticket["error"] = f"script exited with code {process.returncode}"
    except subprocess.TimeoutExpired as error:
        elapsed = round(time.monotonic() - started, 3)
        ticket.update(
            {
                "status": "FAILED",
                "execution_time_sec": elapsed,
                "finished_at": time.time(),
                "error": f"script exceeded timeout of {timeout_seconds:g} seconds",
            }
        )
        if error.stdout or error.stderr:
            result_path = results_directory / f"{ticket_path.stem}_result.txt"
            with result_path.open("w", encoding="utf-8") as output:
                output.write(f"Node: {node_id}\nRuntime: {elapsed}s\n")
                output.write(f"Exit Code: timeout\n\n{error.stdout or ''}")
                if error.stderr:
                    output.write(f"\n[stderr]\n{error.stderr}")
        print(f"[!] Task {ticket_path.stem} timed out")
    except OSError as error:
        ticket.update(
            {
                "status": "FAILED",
                "error": f"task execution failed: {error}",
                "finished_at": time.time(),
            }
        )
        print(f"[!] Could not execute task {ticket_path.stem}: {error}")
    finally:
        ticket["finished_at"] = time.time()
        _write_ticket(ticket_path, ticket)


def run_worker(
    node_id: str,
    task_directory: Path,
    results_directory: Path,
    *,
    poll_interval: float = 1.0,
    timeout_seconds: float = 3600.0,
    stop_event: Optional[threading.Event] = None,
) -> None:
    task_directory.mkdir(parents=True, exist_ok=True)
    print(f"[*] {node_id} polling assigned task tickets in {task_directory}")
    while stop_event is None or not stop_event.is_set():
        for ticket_path in sorted(task_directory.glob("*.json")):
            if ticket_path.stem == "manifest" or ticket_path.name.endswith(
                ".hints.json"
            ):
                continue
            try:
                claim_path = _claim_ticket(ticket_path, timeout_seconds)
            except OSError as error:
                if error.errno in (13, 32):
                    continue
                print(f"[!] Could not claim {ticket_path.name}: {error}")
                continue
            if claim_path is None:
                continue
            ticket = None
            try:
                ticket = _load_ticket(ticket_path)
                if (
                    ticket.get("assigned_to") != node_id
                    or ticket.get("status") != "PENDING"
                ):
                    continue
                dependency_state = _dependencies(ticket, task_directory)
                if dependency_state == "waiting":
                    continue
                if dependency_state == "failed":
                    ticket.update(
                        {
                            "status": "FAILED",
                            "error": "a dependency task failed",
                            "finished_at": time.time(),
                        }
                    )
                    _write_ticket(ticket_path, ticket)
                    continue
                print(f"[+] Running task {ticket_path.stem}")
                _run_ticket(
                    ticket_path,
                    ticket,
                    node_id,
                    timeout_seconds,
                    results_directory,
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                print(f"[!] Could not process {ticket_path.name}: {error}")
                if isinstance(ticket, dict) and ticket.get("status") == "PENDING":
                    ticket.update(
                        {
                            "status": "FAILED",
                            "error": str(error),
                            "finished_at": time.time(),
                        }
                    )
                    try:
                        _write_ticket(ticket_path, ticket)
                    except OSError as write_error:
                        print(
                            f"[!] Could not mark {ticket_path.name} failed: {write_error}"
                        )
            finally:
                try:
                    claim_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError as error:
                    print(f"[!] Could not release task claim {claim_path.name}: {error}")
        if stop_event is None:
            time.sleep(poll_interval)
        else:
            stop_event.wait(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node-id", default=os.environ.get("MARL_NODE_ID", "S1"))
    parser.add_argument(
        "--task-directory",
        type=Path,
        default=Path(
            os.environ.get(
                "MARL_TASKS_DIR", r"\\192.168.137.1\cluster_task"
            )
        ),
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path(
            os.environ.get(
                "MARL_RESULTS_DIR", r"\\192.168.137.1\cluster_result"
            )
        ),
    )
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    arguments = parser.parse_args()
    if arguments.poll_interval <= 0 or arguments.timeout <= 0:
        parser.error("--poll-interval and --timeout must be positive")
    run_worker(
        arguments.node_id,
        arguments.task_directory,
        arguments.results_directory,
        poll_interval=arguments.poll_interval,
        timeout_seconds=arguments.timeout,
    )


if __name__ == "__main__":
    main()
