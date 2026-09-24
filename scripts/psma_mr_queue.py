#!/usr/bin/env python3
"""Durable one-shot continuation watcher for the dsba1 PSMA_MR queue.

The watcher is deliberately independent of the experiment source tree.  It
only waits for the recorded process identity, starts one queued object under a
file lock, and records semantic success/failure from compact result files.
It uses only the Python standard library so it can be copied into an existing
container without changing the active checkout or installing dependencies.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def proc_identity(pid: int) -> dict[str, Any] | None:
    """Read a Linux process identity, treating zombies as terminal.

    ``/proc/<pid>/stat`` puts the comm field in parentheses and it may contain
    spaces, so parsing starts after the final ``") "``.  Field 22 (starttime)
    is offset 19 in the remaining fields.
    """

    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, OSError):
        return None
    try:
        comm, rest = raw.rsplit(") ", 1)
        fields = rest.split()
        state = fields[0]
        starttime = int(fields[19])
        return {
            "pid": pid,
            "comm": comm.removeprefix("(") if comm.startswith("(") else comm,
            "state": state,
            "starttime": str(starttime),
            "live": state != "Z",
        }
    except (ValueError, IndexError):
        return None


def same_process(pid: int, starttime: str | int | None) -> tuple[bool, dict[str, Any] | None]:
    current = proc_identity(pid)
    if current is None or starttime is None:
        return False, current
    return str(current["starttime"]) == str(starttime), current


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_plan(plan: dict[str, Any], *, require_active_identity: bool = True) -> str | None:
    """Validate the narrow v2 queue contract before touching state or process."""

    if plan.get("schema") != 2 or plan.get("revision") != "v2":
        return "queue plan is not schema=2/revision=v2"
    if plan.get("method") != "PSMA_MR":
        return "queue plan method must be PSMA_MR"
    if Path(str(plan.get("config_path", ""))).name != "PSMA_MR_v2.yaml":
        return "queue plan must target configs/methods/PSMA_MR_v2.yaml"
    active = plan.get("active")
    queued = plan.get("queued")
    if not isinstance(active, dict) or not isinstance(queued, dict):
        return "queue plan active/queued sections are required"
    if not active.get("object") or not queued.get("object"):
        return "queue plan active/queued objects are required"
    if active.get("object") == queued.get("object"):
        return "queue plan active and queued objects must differ"
    if require_active_identity:
        try:
            pid = int(active.get("pid"))
        except (TypeError, ValueError):
            return "active process pid is not populated; wait for coordinator launch"
        if pid <= 0 or active.get("starttime") in (None, ""):
            return "active process identity is not populated; wait for coordinator launch"
    for key in ("container", "source_dir", "output_root", "state_path", "lock_path"):
        if not plan.get(key):
            return f"queue plan missing {key}"
    return None


def v2_artifact_identity(config_text: str, manifest: dict[str, Any], object_name: str) -> bool:
    """Reject v1/retry artifacts before a queue item is called succeeded."""

    normalized = " ".join(config_text.lower().split())
    if "method: psma_mr" not in normalized:
        return False
    if "preprocess: no_mask_no_rotation" not in normalized:
        return False
    objects = manifest.get("objects")
    object_meta = objects.get(object_name) if isinstance(objects, dict) else None
    if not isinstance(object_meta, dict) or object_meta.get("method") != "PSMA_MR":
        return False
    if object_meta.get("revision") != "v2":
        return False
    if object_meta.get("feature_layers") != [5, 11, 17, 23]:
        return False
    protocol = object_meta.get("protocol")
    if not isinstance(protocol, dict):
        return False
    if protocol.get("support_count") != 16:
        return False
    context = protocol.get("context")
    training = object_meta.get("training")
    calibration = object_meta.get("calibration")
    return (
        isinstance(context, dict)
        and context.get("masking") is False
        and context.get("rotation") is False
        and isinstance(training, dict)
        and training.get("schedule_full_cartesian") is True
        and isinstance(calibration, dict)
        and calibration.get("clean_normal_only") is True
        and calibration.get("query_used") is False
        and calibration.get("ground_truth_used") is False
    )


def find_final_result(output_dir: Path, object_name: str) -> dict[str, Any] | None:
    """Return the first complete compact result for one expected object."""

    if not output_dir.exists():
        return None
    for metrics_path in sorted(output_dir.rglob("metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        manifest_path = run_dir / "run_manifest.json"
        if not config_path.is_file() or not manifest_path.is_file():
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config_text = config_path.read_text(encoding="utf-8")
        except (OSError, ValueError, TypeError):
            continue
        metrics = payload.get(object_name)
        objects = manifest.get("objects")
        if not isinstance(metrics, dict) or not isinstance(objects, dict):
            continue
        if object_name not in objects:
            continue
        if not v2_artifact_identity(config_text, manifest, object_name):
            continue
        if not finite_number(metrics.get("seg_AUPRO_0.05")):
            continue
        if not finite_number(metrics.get("seg_AUROC_0.05")):
            continue
        return {
            "metrics": str(metrics_path),
            "config": str(config_path),
            "manifest": str(manifest_path),
            "metrics_payload": metrics,
        }
    return None


def initial_state(plan: dict[str, Any]) -> dict[str, Any]:
    active = plan["active"]
    queued = plan["queued"]
    return {
        "schema": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "watcher_pid": os.getpid(),
        "watcher_starttime": proc_identity(os.getpid()),
        "gpu_label": plan["gpu_label"],
        "container": plan["container"],
        "source_dir": plan["source_dir"],
        "output_root": plan["output_root"],
        "active": {
            "object": active["object"],
            "pid": int(active["pid"]),
            "starttime": str(active["starttime"]),
            "status": "waiting",
        },
        "queued": {
            "object": queued["object"],
            "status": "pending",
            "pid": None,
            "starttime": None,
            "log": None,
        },
        "terminal": False,
    }


def append_event(state: dict[str, Any], event: str, **fields: Any) -> None:
    events = state.setdefault("events", [])
    events.append({"at": utc_now(), "event": event, **fields})
    # A queue state is intentionally compact; retain enough history for audit
    # without growing indefinitely if a process repeatedly changes state.
    if len(events) > 32:
        del events[:-32]


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now()
    atomic_json(path, state)


def result_status(plan: dict[str, Any], object_name: str) -> tuple[str, dict[str, Any] | None]:
    result = find_final_result(Path(plan["output_root"]) / plan["output_dir_name"], object_name)
    return ("succeeded", result) if result else ("failed", None)


def output_dir_for(plan: dict[str, Any], item: dict[str, Any]) -> Path:
    explicit = item.get("output_dir")
    if explicit:
        return Path(str(explicit))
    return Path(plan["output_root"]) / str(item["output_dir_name"])


def semantic_result(plan: dict[str, Any], item: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    output_dir = output_dir_for(plan, item)
    result = find_final_result(output_dir, str(item["object"]))
    if result:
        return "succeeded", result
    return "failed", None


def launch_queued(plan: dict[str, Any], state: dict[str, Any], log_path: Path) -> None:
    queued = plan["queued"]
    command = [str(value) for value in plan["command"]]
    source_dir = str(plan["source_dir"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("ab", buffering=0)
    try:
        stream.write(f"[{utc_now()}] launch: {json.dumps(command)}\n".encode())
        process = subprocess.Popen(
            command,
            cwd=source_dir,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        stream.close()
    starttime = None
    for _ in range(20):
        identity = proc_identity(process.pid)
        if identity is not None:
            starttime = identity["starttime"]
            break
        time.sleep(0.05)
    queued_state = state["queued"]
    queued_state.update(
        {
            "status": "started",
            "pid": process.pid,
            "starttime": starttime,
            "log": str(log_path),
            "started_at": utc_now(),
        }
    )
    append_event(state, "queued_started", object=queued["object"], pid=process.pid, starttime=starttime)


def finalize_item(
    plan: dict[str, Any], state: dict[str, Any], section: str, item: dict[str, Any]
) -> str:
    status, result = semantic_result(plan, item)
    target = state[section]
    target["status"] = status
    target["finished_at"] = utc_now()
    target["result"] = result
    append_event(state, f"{section}_finished", object=item["object"], status=status, result=result)
    return status


def watcher(plan_path: Path, poll_seconds: float) -> int:
    plan = read_json(plan_path)
    if plan is None:
        print(f"invalid plan: {plan_path}", file=sys.stderr)
        return 2
    validation_error = validate_plan(plan)
    if validation_error:
        print(validation_error, file=sys.stderr)
        return 2
    state_path = Path(str(plan["state_path"]))
    lock_path = Path(str(plan["lock_path"]))
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("queue watcher already active (lock held)")
            return 0

        state = read_json(state_path)
        if state is None:
            state = initial_state(plan)
            save_state(state_path, state)
        elif state.get("terminal"):
            print("queue already terminal")
            return 0

        # Existing state is authoritative.  A restarted watcher never relaunches
        # a queued object whose state is already started but whose PID vanished.
        queued_state = state.setdefault("queued", {})
        if queued_state.get("status") == "started":
            pid = int(queued_state.get("pid") or 0)
            matches, identity = same_process(pid, queued_state.get("starttime"))
            if not matches or not identity or not identity.get("live", False):
                finalize_item(plan, state, "queued", plan["queued"])
                state["terminal"] = True
                append_event(state, "watcher_finished", reason="restarted_after_queued_exit")
                save_state(state_path, state)
                print(f"queued terminal: {state['queued']['status']}")
                return 0

        while True:
            active_state = state["active"]
            if active_state.get("status") in {"waiting", "running"}:
                pid = int(active_state["pid"])
                matches, identity = same_process(pid, active_state.get("starttime"))
                if matches and identity and identity.get("live", False):
                    active_state["status"] = "running"
                    save_state(state_path, state)
                    time.sleep(poll_seconds)
                    continue
                finalize_item(plan, state, "active", plan["active"])
                save_state(state_path, state)

            queued_state = state["queued"]
            if queued_state.get("status") == "pending":
                log_path = Path(str(plan["queued_log_path"]))
                launch_queued(plan, state, log_path)
                save_state(state_path, state)
                continue

            if queued_state.get("status") == "started":
                pid = int(queued_state.get("pid") or 0)
                matches, identity = same_process(pid, queued_state.get("starttime"))
                if matches and identity and identity.get("live", False):
                    time.sleep(poll_seconds)
                    continue
                finalize_item(plan, state, "queued", plan["queued"])
                state["terminal"] = True
                append_event(state, "watcher_finished", reason="queued_exit")
                save_state(state_path, state)
                print(f"queued terminal: {state['queued']['status']}")
                return 0

            if queued_state.get("status") in {"succeeded", "failed"}:
                state["terminal"] = True
                save_state(state_path, state)
                return 0
            print("unexpected queue state", file=sys.stderr)
            return 2


def status(plan_path: Path) -> int:
    plan = read_json(plan_path)
    if plan is None:
        print(f"invalid plan: {plan_path}", file=sys.stderr)
        return 2
    validation_error = validate_plan(plan, require_active_identity=False)
    if validation_error:
        print(validation_error, file=sys.stderr)
        return 2
    state = read_json(Path(str(plan["state_path"])))
    if state is None:
        print("state=missing")
        return 1
    print(json.dumps(state, sort_keys=True))
    return 0


def bind_plan(plan_path: Path, pid: int, starttime: str, host_pid: int | None) -> int:
    """Record the coordinator-launched active process identity atomically."""

    plan = read_json(plan_path)
    if plan is None:
        print(f"invalid plan: {plan_path}", file=sys.stderr)
        return 2
    validation_error = validate_plan(plan, require_active_identity=False)
    if validation_error:
        print(validation_error, file=sys.stderr)
        return 2
    if pid <= 0 or not str(starttime):
        print("active process pid/starttime are required", file=sys.stderr)
        return 2
    active = plan["active"]
    active.update(
        {
            "pid": int(pid),
            "starttime": str(starttime),
            "host_pid": int(host_pid) if host_pid is not None else None,
            "status": "started",
            "bound_at": utc_now(),
        }
    )
    atomic_json(plan_path, plan)
    print(json.dumps(active, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    watch_parser = sub.add_parser("watch")
    watch_parser.add_argument("--plan", required=True, type=Path)
    watch_parser.add_argument("--poll-seconds", type=float, default=30.0)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--plan", required=True, type=Path)
    bind_parser = sub.add_parser("bind")
    bind_parser.add_argument("--plan", required=True, type=Path)
    bind_parser.add_argument("--pid", required=True, type=int)
    bind_parser.add_argument("--starttime", required=True)
    bind_parser.add_argument("--host-pid", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "watch":
        return watcher(args.plan, max(float(args.poll_seconds), 1.0))
    if args.command == "status":
        return status(args.plan)
    if args.command == "bind":
        return bind_plan(args.plan, args.pid, args.starttime, args.host_pid)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
