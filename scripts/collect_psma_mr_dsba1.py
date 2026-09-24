#!/usr/bin/env python3
"""Durable compact collector/report for the dsba1 PSMA_MR eight-class queue.

Only ``metrics.json``, ``config.yaml``, and ``run_manifest.json`` are copied
from the existing containers.  The collector never pulls logs, maps, weights,
or checkpoints; it writes bounded progress reports and appends newly verified
class rows incrementally, with a macro row only after every queue watcher is
terminal and all eight classes are verified.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
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

try:
    import yaml
except ImportError:  # pragma: no cover - a missing project dependency is an invalid run
    yaml = None


ARTIFACT_NAMES = ("metrics.json", "config.yaml", "run_manifest.json")
METRIC_NAMES = ("seg_AUPRO_0.05", "seg_AUROC_0.05")
CANONICAL_FIELDS = [
    "row_type",
    "method",
    "scope",
    "class_name",
    "n_classes",
    "backbone",
    "preprocess",
    "split",
    "support_shots",
    "support_policy",
    "support_identity",
    "seed",
    "training_or_adapter",
    "resolution",
    "evaluator_metric_contract",
    "seg_AUPRO_0.05",
    "seg_AUROC_0.05",
    "delta_AUPRO_vs_SuperAD",
    "delta_AUROC_vs_SuperAD",
    "comparison_status",
    "source",
]


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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def local_process_starttime(pid: int) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        _, rest = raw.rsplit(") ", 1)
        return str(int(rest.split()[19]))
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def parse_yaml_mapping(text: str) -> dict[str, Any] | None:
    """Parse the persisted resolved config using the project's YAML schema.

    The previous collector searched normalized YAML text.  That made a valid
    multiline list such as ``resolutions:\n  - 448`` look different from the
    inline form and allowed schema drift to be hidden by ad-hoc string checks.
    A parse failure remains unverified rather than being treated as success.
    """

    if yaml is None:
        return None
    try:
        value = yaml.safe_load(text)
    except (AttributeError, TypeError, ValueError, yaml.YAMLError):
        return None
    return value if isinstance(value, dict) else None


def nonempty_hash(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def nested_mapping(value: Any, key: str) -> dict[str, Any]:
    child = value.get(key) if isinstance(value, dict) else None
    return child if isinstance(child, dict) else {}


def plan_specs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    specs = plan.get("specs")
    if (
        not isinstance(specs, list)
        or len(specs) != 4
        or any(not isinstance(spec, dict) for spec in specs)
    ):
        raise ValueError("collector plan must contain exactly four queue specs")
    return specs


def ssh_base(plan: dict[str, Any]) -> list[str]:
    host = os.environ.get("REMOTE_HOST", "")
    if not host:
        raise RuntimeError("REMOTE_HOST must be present in the collector environment")
    args = ["ssh", "-p", os.environ.get("REMOTE_PORT", "22"), "-o", "ConnectTimeout=10"]
    jump = os.environ.get("JUMP_HOST", "")
    if jump:
        args.extend(["-J", f"{jump}:{os.environ.get('JUMP_PORT', '22')}"])
    args.append(host)
    return args


def ssh_run(plan: dict[str, Any], *remote_args: str, timeout: float = 30.0) -> str | None:
    command = ssh_base(plan) + [str(arg) for arg in remote_args]
    environment = os.environ.copy()
    password = environment.get("SSH_PASSWORD")
    if password:
        environment.setdefault("DISPLAY", "dummy")
        # The credential env file may define SSH_ASKPASS as an empty value;
        # force the project-local helper rather than inheriting that value.
        environment["SSH_ASKPASS"] = str(Path(__file__).with_name("ssh_askpass_env.sh"))
        environment["SSH_ASKPASS_REQUIRE"] = "force"
        command = ["setsid", "-w", *command]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def remote_files(plan: dict[str, Any], spec: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    container = str(spec["container"])
    for output_dir_key in ("active_output_dir", "queued_output_dir"):
        output_dir = str(spec[output_dir_key])
        for name in ARTIFACT_NAMES:
            output = ssh_run(
                plan,
                "docker",
                "exec",
                container,
                "find",
                output_dir,
                "-type",
                "f",
                "-name",
                name,
                "-print",
            )
            if output:
                found.update(line.strip() for line in output.splitlines() if line.strip())
    return sorted(found)


def remote_text(plan: dict[str, Any], container: str, path: str) -> str | None:
    return ssh_run(plan, "docker", "exec", container, "cat", path, timeout=30.0)


def local_artifact_path(root: Path, spec: dict[str, Any], remote_path: str) -> Path:
    class_names = {
        Path(str(spec["active_output_dir"])).name,
        Path(str(spec["queued_output_dir"])).name,
    }
    parts = Path(remote_path).parts
    class_index = next(
        (index for index, part in enumerate(parts) if part in class_names),
        len(parts) - 1,
    )
    relative_path = Path(*parts[class_index:])
    safe_parts = [part for part in relative_path.parts if part not in {"", ".", ".."}]
    return root / str(spec["gpu_label"]) / Path(*safe_parts)


def recover_artifacts(plan: dict[str, Any], spec: dict[str, Any], root: Path) -> list[Path]:
    recovered: list[Path] = []
    container = str(spec["container"])
    for remote_path in remote_files(plan, spec):
        name = Path(remote_path).name
        if name not in ARTIFACT_NAMES:
            continue
        content = remote_text(plan, container, remote_path)
        if content is None:
            continue
        target = local_artifact_path(root, spec, remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        recovered.append(target)
    return recovered


def condition_checks(
    config: dict[str, Any],
    manifest: dict[str, Any],
    metrics_payload: dict[str, Any],
    object_name: str,
    expected: dict[str, Any],
    expected_output_dir: str,
) -> dict[str, bool]:
    """Validate the resolved v2 run contract against actual artifact schema.

    ``run_manifest.json`` intentionally stores query fingerprints/count rather
    than every query path.  When paths are present we inspect them directly;
    otherwise the count and all fingerprints must be present and the configured
    pipeline split must be the known ``test_public`` default.  This keeps the
    check evidence-backed without rejecting the current manifest schema.
    """

    method_config = nested_mapping(config, "method_config")
    backbone_config = nested_mapping(config, "backbone_config")
    dataset_config = nested_mapping(config, "dataset_config")
    protocol_config = nested_mapping(config, "protocol")
    evaluation = nested_mapping(config, "evaluation")
    output = nested_mapping(config, "output")
    objects = manifest.get("objects")
    object_meta = objects.get(object_name) if isinstance(objects, dict) else None
    object_meta = object_meta if isinstance(object_meta, dict) else {}
    manifest_protocol = nested_mapping(object_meta, "protocol")
    backbone = nested_mapping(object_meta, "backbone")
    context = nested_mapping(manifest_protocol, "context")
    support_paths = manifest_protocol.get("support_paths")
    query_paths = manifest_protocol.get("query_paths")
    query_count = manifest_protocol.get("query_count")

    expected_method = str(expected.get("method", "PSMA_MR"))
    expected_dataset = str(expected.get("dataset", "mvtec_ad2"))
    expected_backbone = str(expected.get("backbone", "dinov2_vitl14"))
    expected_seed = expected.get("seed", 0)
    expected_shots = expected.get("shots", 16)
    expected_preprocess = str(expected.get("preprocess", "no_mask_no_rotation"))
    expected_metric = str(
        expected.get("metric_contract", "mvtec_ad2_float16_streaming_v1")
    )
    expected_revision = str(expected.get("revision", "v2"))
    expected_feature_layers = expected.get("feature_layers", [5, 11, 17, 23])
    expected_split = str(expected.get("query_split", "test_public"))

    resolutions = method_config.get("resolutions")
    expected_resolutions = expected.get("resolutions")
    resolutions_ok = isinstance(resolutions, list) and bool(resolutions)
    if expected_resolutions is not None:
        resolutions_ok = resolutions == expected_resolutions

    if isinstance(query_paths, list):
        query_split_ok = bool(query_paths) and all(
            "/test_public/" in str(path).replace("\\", "/") for path in query_paths
        )
    else:
        # The current pipeline records query_count and fingerprints, not paths.
        # Requiring all fingerprints prevents an absent path list from becoming
        # an unconditional pass.
        query_split_ok = expected_split == "test_public" and isinstance(query_count, int) and query_count > 0

    config_output_dir = config.get("output_dir")
    output_dir_ok = (
        isinstance(config_output_dir, str)
        and Path(config_output_dir).as_posix() == Path(expected_output_dir).as_posix()
    )
    query_fingerprints_ok = all(
        nonempty_hash(manifest_protocol.get(name))
        for name in ("query_identity_sha256", "query_content_sha256", "ground_truth_sha256")
    )
    context_v2_ok = context.get("masking") is False and context.get("rotation") is False
    manifest_method_ok = object_meta.get("method") == expected_method
    training_meta = nested_mapping(object_meta, "training")
    calibration_meta = nested_mapping(object_meta, "calibration")
    manifest_object_ok = isinstance(objects, dict) and object_name in objects
    config_objects = dataset_config.get("objects")
    object_config_ok = (
        isinstance(config_objects, list)
        and object_name in config_objects
    )

    config_checks = {
        "method": config.get("method") == expected_method,
        "dataset": config.get("dataset") == expected_dataset,
        "backbone_config": config.get("backbone") == expected_backbone,
        "revision": (
            method_config.get("revision") == expected_revision
            and object_meta.get("revision") == expected_revision
        ),
        "feature_layers": (
            backbone_config.get("feature_layers") == expected_feature_layers
            and object_meta.get("feature_layers") == expected_feature_layers
        ),
        "seed": config.get("seed") == expected_seed,
        "shots": protocol_config.get("shots") == expected_shots,
        "preprocess": dataset_config.get("preprocess") == expected_preprocess,
        "resolutions": resolutions_ok,
        "maps_off": output.get("save_anomaly_maps") is False,
        "weights_off": output.get("save_weights") is False,
        "metric_contract": (
            evaluation.get("metric_contract") == expected_metric
            and metrics_payload.get("metric_contract") == expected_metric
        ),
        "output_dir": output_dir_ok,
        "object_configured": object_config_ok,
    }
    protocol_checks = {
        "object": manifest_object_ok,
        "manifest_method": manifest_method_ok,
        "support_16": manifest_protocol.get("support_count") == expected_shots
        and isinstance(support_paths, list)
        and len(support_paths) == expected_shots,
        "test_public": query_split_ok,
        "query_fingerprints": query_fingerprints_ok,
        "backbone": backbone.get("name") == expected_backbone,
        "context_no_mask_no_rotation": context_v2_ok,
        "schedule_full_cartesian": training_meta.get("schedule_full_cartesian") is True,
        "calibration_query_free": (
            calibration_meta.get("clean_normal_only") is True
            and calibration_meta.get("query_used") is False
            and calibration_meta.get("ground_truth_used") is False
        ),
        "v2_identity": (
            config.get("method") == expected_method
            and dataset_config.get("preprocess") == expected_preprocess
            and context_v2_ok
            and output_dir_ok
        ),
    }
    metric_checks = {
        name: finite_number(metrics_payload.get(name)) for name in METRIC_NAMES
    }
    return {**config_checks, **protocol_checks, **metric_checks}


def load_verified_result(
    spec: dict[str, Any], recovered_root: Path, expected: dict[str, Any]
) -> tuple[dict[str, Any] | None, str]:
    object_name = str(spec["object"])
    candidates = sorted(
        path for path in (recovered_root / str(spec["gpu_label"])).rglob("metrics.json")
    )
    for metrics_path in candidates:
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        manifest_path = run_dir / "run_manifest.json"
        if not config_path.is_file() or not manifest_path.is_file():
            continue
        metrics_payload = read_json(metrics_path)
        manifest = read_json(manifest_path)
        if not isinstance(metrics_payload, dict) or not isinstance(manifest, dict):
            continue
        metrics = metrics_payload.get(object_name)
        if not isinstance(metrics, dict):
            continue
        try:
            config = parse_yaml_mapping(config_path.read_text(encoding="utf-8"))
        except OSError:
            config = None
        if config is None:
            return (
                {
                    "result_status": "unverified",
                    "condition_checks": {"config_parse": False},
                    "metrics_path": str(metrics_path),
                    "config_path": str(config_path),
                    "manifest_path": str(manifest_path),
                },
                "resolved config is not valid YAML or PyYAML is unavailable",
            )
        checks = condition_checks(
            config,
            manifest,
            metrics,
            object_name,
            expected,
            str(spec["output_dir"]),
        )
        if not all(checks.values()):
            return (
                {
                    "result_status": "unverified",
                    "condition_checks": checks,
                    "metrics_path": str(metrics_path),
                    "config_path": str(config_path),
                    "manifest_path": str(manifest_path),
                },
                "metric or split/support/model conditions are not fully verified",
            )
        return (
            {
                "result_status": "succeeded",
                "condition_checks": checks,
                "metrics_path": str(metrics_path),
                "config_path": str(config_path),
                "manifest_path": str(manifest_path),
                "seg_AUPRO_0.05": metrics["seg_AUPRO_0.05"],
                "seg_AUROC_0.05": metrics["seg_AUROC_0.05"],
                "run_id": manifest.get("run_id", "N/A"),
                "config_hash": manifest.get("config_hash", "N/A"),
                "code_sha256": manifest.get("code_sha256", "N/A"),
                "support_identity_sha256": nested_mapping(
                    nested_mapping(manifest.get("objects"), object_name), "protocol"
                ).get("support_identity_sha256", "N/A"),
                "query_identity_sha256": nested_mapping(
                    nested_mapping(manifest.get("objects"), object_name), "protocol"
                ).get("query_identity_sha256", "N/A"),
                "query_count": nested_mapping(
                    nested_mapping(manifest.get("objects"), object_name), "protocol"
                ).get("query_count", "N/A"),
                "resolution": "+".join(
                    str(value)
                    for value in nested_mapping(config, "method_config").get(
                        "resolutions", []
                    )
                )
                or "N/A",
                "fit_seconds": "N/A",
                "inference_seconds": "N/A",
            },
            "verified PSMA_MR-v2/Testpublic/support-16/no-mask-no-rotation/DINOv2 conditions; native metric semantics",
        )
    return None, "final compact metrics/config/manifest not present"


def state_for(plan: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    text = remote_text(plan, str(spec["container"]), str(spec["state_path"]))
    if not text:
        return None
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def item_terminal_status(state: dict[str, Any] | None, spec: dict[str, Any], item: str) -> str:
    if state is None:
        return "unknown"
    section = state.get(item)
    if not isinstance(section, dict):
        return "unknown"
    value = section.get("status")
    if value in {"succeeded", "failed"}:
        return str(value)
    return str(value or "pending")


def collect_once(plan: dict[str, Any]) -> dict[str, Any]:
    local_root = Path(str(plan["local_root"]))
    recovered_root = local_root / "recovered"
    expected = plan.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("collector plan must define expected v2 conditions")
    rows: list[dict[str, Any]] = []
    all_terminal = True
    for spec in plan_specs(plan):
        state = state_for(plan, spec)
        queue_terminal = bool(state and state.get("terminal") is True)
        all_terminal = all_terminal and queue_terminal
        recover_artifacts(plan, spec, recovered_root)
        for item_name, item_key in (
            (str(spec["active_object"]), "active"),
            (str(spec["queued_object"]), "queued"),
        ):
            item_spec = dict(spec)
            item_spec["object"] = item_name
            item_spec["output_dir"] = str(spec[f"{item_key}_output_dir"])
            result, note = load_verified_result(item_spec, recovered_root, expected)
            terminal_status = item_terminal_status(state, spec, item_key)
            if result is None:
                row = {
                    "class": item_name,
                    "gpu": spec["gpu_label"],
                    "host_gpu": spec["host_gpu"],
                    "container": spec["container"],
                    "queue_status": terminal_status,
                    "result_status": (
                        "failed" if terminal_status == "failed" else "running"
                    ),
                    "seg_AUPRO_0.05": "N/A",
                    "seg_AUROC_0.05": "N/A",
                    "resolution": "N/A",
                    "run_id": "N/A",
                    "config_hash": "N/A",
                    "code_sha256": "N/A",
                    "support_identity_sha256": "N/A",
                    "query_identity_sha256": "N/A",
                    "query_count": "N/A",
                    "fit_seconds": "N/A",
                    "inference_seconds": "N/A",
                    "condition_status": "N/A",
                    "comparison": "PSMA_MR class row only; no verified baseline match",
                    "note": note,
                    "metrics_path": "N/A",
                    "config_path": "N/A",
                    "manifest_path": "N/A",
                }
            else:
                row = {
                    "class": item_name,
                    "gpu": spec["gpu_label"],
                    "host_gpu": spec["host_gpu"],
                    "container": spec["container"],
                    "queue_status": terminal_status,
                    "result_status": result["result_status"],
                    "seg_AUPRO_0.05": result.get("seg_AUPRO_0.05", "N/A"),
                    "seg_AUROC_0.05": result.get("seg_AUROC_0.05", "N/A"),
                    "resolution": result.get("resolution", "N/A"),
                    "run_id": result.get("run_id", "N/A"),
                    "config_hash": result.get("config_hash", "N/A"),
                    "code_sha256": result.get("code_sha256", "N/A"),
                    "support_identity_sha256": result.get(
                        "support_identity_sha256", "N/A"
                    ),
                    "query_identity_sha256": result.get(
                        "query_identity_sha256", "N/A"
                    ),
                    "query_count": result.get("query_count", "N/A"),
                    "fit_seconds": result.get("fit_seconds", "N/A"),
                    "inference_seconds": result.get("inference_seconds", "N/A"),
                    "condition_status": (
                        "verified" if result["result_status"] == "succeeded" else "unverified"
                    ),
                    "comparison": "PSMA_MR class row only; no verified baseline match",
                    "note": note,
                    "metrics_path": result.get("metrics_path", "N/A"),
                    "config_path": result.get("config_path", "N/A"),
                    "manifest_path": result.get("manifest_path", "N/A"),
                }
            rows.append(row)
    return {"timestamp": utc_now(), "terminal": all_terminal, "rows": rows}


def write_reports(plan: dict[str, Any], snapshot: dict[str, Any]) -> None:
    rows = snapshot["rows"]
    local_root = Path(str(plan["local_root"]))
    csv_path = Path(str(plan["report_csv"]))
    md_path = Path(str(plan["report_md"]))
    fields = [
        "class",
        "gpu",
        "host_gpu",
        "container",
        "queue_status",
        "result_status",
        "seg_AUPRO_0.05",
        "seg_AUROC_0.05",
        "resolution",
        "run_id",
        "config_hash",
        "code_sha256",
        "support_identity_sha256",
        "query_identity_sha256",
        "query_count",
        "fit_seconds",
        "inference_seconds",
        "condition_status",
        "comparison",
        "note",
        "metrics_path",
        "config_path",
        "manifest_path",
    ]
    valid = [
        row
        for row in rows
        if row["result_status"] == "succeeded"
        and row["condition_status"] == "verified"
        and finite_number(row["seg_AUPRO_0.05"])
        and finite_number(row["seg_AUROC_0.05"])
    ]
    full8 = len(rows) == 8 and len(valid) == 8
    macro_aupro = sum(float(row["seg_AUPRO_0.05"]) for row in valid) / 8 if full8 else "N/A"
    macro_auroc = sum(float(row["seg_AUROC_0.05"]) for row in valid) / 8 if full8 else "N/A"
    macro = {
        "class": "MACRO_MEAN_FULL8_ONLY",
        "gpu": "N/A",
        "host_gpu": "N/A",
        "container": "N/A",
        "queue_status": "terminal",
        "result_status": "full8" if full8 else "not_computable",
        "seg_AUPRO_0.05": macro_aupro,
        "seg_AUROC_0.05": macro_auroc,
        "resolution": "N/A",
        "run_id": "N/A",
        "config_hash": "N/A",
        "code_sha256": "N/A",
        "support_identity_sha256": "N/A",
        "query_identity_sha256": "N/A",
        "query_count": "N/A",
        "fit_seconds": "N/A",
        "inference_seconds": "N/A",
        "condition_status": "full8_verified" if full8 else "N/A",
        "comparison": "No partial macro mean",
        "note": "Full-eight mean only; missing/failed/unverified classes keep this N/A.",
        "metrics_path": "N/A",
        "config_path": "N/A",
        "manifest_path": "N/A",
    }
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=csv_path.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow(macro)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, csv_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def canonical_number(value: Any) -> str:
    if finite_number(value):
        return f"{float(value):.17g}"
    return str(value)


def canonical_delta(value: Any, baseline: Any) -> str:
    if not finite_number(value) or not finite_number(baseline):
        return "N/A"
    return f"{float(value) - float(baseline):.17g}"


def run_identity(row: dict[str, Any]) -> str:
    config_hash = str(row.get("config_hash", "N/A"))
    run_id = str(row.get("run_id", "N/A"))
    return f"v2_config_hash={config_hash};run_id={run_id}"


def append_canonical_ledger(plan: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Append only verified v2 evidence to the live cumulative ledger.

    Existing rows are read and written back unchanged.  New rows carry the
    config/run identity in ``training_or_adapter`` and use a distinct scope,
    so a rerun cannot collide with the historical PSMA_MR rows or become a
    false full8 result.
    """

    csv_path = Path(str(plan.get("canonical_csv", "")))
    md_path = Path(str(plan.get("canonical_md", "")))
    if not csv_path or not md_path:
        raise ValueError("collector plan must define canonical_csv and canonical_md")
    if not csv_path.is_file() or not md_path.is_file():
        raise FileNotFoundError("canonical comparison ledger is missing")

    expected = plan.get("expected")
    if not isinstance(expected, dict):
        raise ValueError("collector plan must define expected v2 conditions")
    rows = snapshot.get("rows")
    if not isinstance(rows, list):
        raise ValueError("collector snapshot rows are invalid")
    valid = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("result_status") == "succeeded"
        and row.get("condition_status") == "verified"
        and finite_number(row.get("seg_AUPRO_0.05"))
        and finite_number(row.get("seg_AUROC_0.05"))
    ]
    if not valid:
        return {
            "status": "no_verified_rows",
            "appended": 0,
            "scope": "dsba1_v2_partial",
            "csv": str(csv_path),
            "markdown": str(md_path),
        }

    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CANONICAL_FIELDS:
            raise ValueError("canonical comparison CSV schema changed unexpectedly")
        existing = list(reader)
    if len(existing) < 26:
        raise ValueError("canonical ledger has fewer than the preserved 26 history rows")

    baseline_by_class = {
        row.get("class_name"): row
        for row in existing
        if row.get("method") == "SuperAD-16" and row.get("row_type") == "class"
    }
    baseline_macro = next(
        (
            row
            for row in existing
            if row.get("method") == "SuperAD-16" and row.get("row_type") == "macro"
        ),
        {},
    )
    full8 = len(rows) == 8 and len(valid) == 8
    scope = "dsba1_v2_full8" if full8 else "dsba1_v2_partial"
    method = str(expected.get("ledger_method", "PSMA_MR_v2"))
    metric_contract = str(expected.get("metric_contract", "mvtec_ad2_float16_streaming_v1"))
    append_rows: list[dict[str, Any]] = []
    for row in valid:
        baseline = baseline_by_class.get(str(row["class"])) or {}
        append_rows.append(
            {
                "row_type": "class",
                "method": method,
                "scope": scope,
                "class_name": str(row["class"]),
                "n_classes": "8",
                "backbone": str(expected.get("backbone", "dinov2_vitl14")),
                "preprocess": str(expected.get("preprocess", "no_mask_no_rotation")),
                "split": str(expected.get("query_split", "test_public")),
                "support_shots": str(expected.get("shots", 16)),
                "support_policy": "random16",
                "support_identity": str(row.get("support_identity_sha256", "N/A")),
                "seed": str(expected.get("seed", 0)),
                "training_or_adapter": run_identity(row),
                "resolution": str(row.get("resolution", "N/A")),
                "evaluator_metric_contract": metric_contract,
                "seg_AUPRO_0.05": canonical_number(row["seg_AUPRO_0.05"]),
                "seg_AUROC_0.05": canonical_number(row["seg_AUROC_0.05"]),
                "delta_AUPRO_vs_SuperAD": canonical_delta(
                    row["seg_AUPRO_0.05"], baseline.get("seg_AUPRO_0.05")
                ),
                "delta_AUROC_vs_SuperAD": canonical_delta(
                    row["seg_AUROC_0.05"], baseline.get("seg_AUROC_0.05")
                ),
                "comparison_status": "numeric_historical_comparison_v2_no_new_baseline",
                "source": str(row.get("metrics_path", "N/A")),
            }
        )

    run_tokens = sorted(
        f"{row['class']}:{row.get('config_hash', 'N/A')}:{row.get('run_id', 'N/A')}"
        for row in valid
    )
    run_set_hash = hashlib.sha256("|".join(run_tokens).encode()).hexdigest()
    if full8:
        macro_aupro = math.fsum(float(row["seg_AUPRO_0.05"]) for row in valid) / 8
        macro_auroc = math.fsum(float(row["seg_AUROC_0.05"]) for row in valid) / 8
        append_rows.append(
            {
                "row_type": "macro",
                "method": method,
                "scope": scope,
                "class_name": "ALL",
                "n_classes": "8",
                "backbone": str(expected.get("backbone", "dinov2_vitl14")),
                "preprocess": str(expected.get("preprocess", "no_mask_no_rotation")),
                "split": str(expected.get("query_split", "test_public")),
                "support_shots": str(expected.get("shots", 16)),
                "support_policy": "random16",
                "support_identity": "class_specific_manifest_set",
                "seed": str(expected.get("seed", 0)),
                "training_or_adapter": f"v2_run_set_sha256={run_set_hash}",
                "resolution": "N/A",
                "evaluator_metric_contract": metric_contract,
                "seg_AUPRO_0.05": canonical_number(macro_aupro),
                "seg_AUROC_0.05": canonical_number(macro_auroc),
                "delta_AUPRO_vs_SuperAD": canonical_delta(
                    macro_aupro, baseline_macro.get("seg_AUPRO_0.05")
                ),
                "delta_AUROC_vs_SuperAD": canonical_delta(
                    macro_auroc, baseline_macro.get("seg_AUROC_0.05")
                ),
                "comparison_status": "numeric_historical_comparison_v2_no_new_baseline",
                "source": str(Path(str(plan["local_root"])) / "recovered"),
            }
        )

    identity_keys = {
        (
            row.get("method"),
            row.get("scope"),
            row.get("class_name"),
            row.get("seed"),
            row.get("training_or_adapter"),
        )
        for row in existing
    }
    new_rows = [
        row
        for row in append_rows
        if (
            row["method"],
            row["scope"],
            row["class_name"],
            row["seed"],
            row["training_or_adapter"],
        )
        not in identity_keys
    ]
    if not new_rows:
        return {
            "status": "already_present",
            "appended": 0,
            "scope": scope,
            "run_set_sha256": run_set_hash,
            "csv": str(csv_path),
            "markdown": str(md_path),
        }

    lock_path = Path(str(plan["local_root"])) / ".canonical.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        # Re-read after acquiring the lock in case another collector completed
        # between the initial read and this append.
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != CANONICAL_FIELDS:
                raise ValueError("canonical comparison CSV schema changed unexpectedly")
            existing = list(reader)
        existing_keys = {
            (
                row.get("method"),
                row.get("scope"),
                row.get("class_name"),
                row.get("seed"),
                row.get("training_or_adapter"),
            )
            for row in existing
        }
        new_rows = [
            row
            for row in append_rows
            if (
                row["method"],
                row["scope"],
                row["class_name"],
                row["seed"],
                row["training_or_adapter"],
            )
            not in existing_keys
        ]
        if new_rows:
            fd, temporary = tempfile.mkstemp(prefix=f".{csv_path.name}.", dir=csv_path.parent)
            try:
                with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=CANONICAL_FIELDS)
                    writer.writeheader()
                    writer.writerows(existing)
                    writer.writerows(new_rows)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, csv_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

        markdown = md_path.read_text(encoding="utf-8")
        marker = f"## PSMA_MR_v2 dsba1 ({run_set_hash})"
        if new_rows and marker not in markdown:
            section = [
                "",
                marker,
                "",
                f"Scope: `{scope}`; method identity: `{method}`; seed `{expected.get('seed', 0)}`; support `random16`; split `{expected.get('query_split', 'test_public')}`; preprocess `{expected.get('preprocess', 'no_mask_no_rotation')}`.",
                "This section appends only verified v2 artifacts. No new baseline was executed; deltas are numeric comparisons against preserved SuperAD-16 rows. Missing/failed classes are not assigned metric values.",
                "",
                "| Class | AUPRO_0.05 | AUROC_0.05 | Δ AUPRO vs SuperAD | Δ AUROC vs SuperAD | Config/run identity |",
                "|---|---:|---:|---:|---:|---|",
            ]
            for row in new_rows:
                if row["row_type"] != "class":
                    continue
                section.append(
                    f"| {row['class_name']} | {row['seg_AUPRO_0.05']} | {row['seg_AUROC_0.05']} | {row['delta_AUPRO_vs_SuperAD']} | {row['delta_AUROC_vs_SuperAD']} | {row['training_or_adapter']} |"
                )
            macro_rows = [row for row in new_rows if row["row_type"] == "macro"]
            if macro_rows:
                macro = macro_rows[0]
                section.extend(
                    [
                        "",
                        f"**Macro mean (full8 only):** AUPRO_0.05={macro['seg_AUPRO_0.05']}, AUROC_0.05={macro['seg_AUROC_0.05']}; run-set `{run_set_hash}`.",
                    ]
                )
            markdown += "\n".join(section) + "\n"
            fd, temporary = tempfile.mkstemp(prefix=f".{md_path.name}.", dir=md_path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(markdown)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, md_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    return {
        "status": "appended" if new_rows else "already_present",
        "appended": len(new_rows),
        "scope": scope,
        "run_set_sha256": run_set_hash,
        "csv": str(csv_path),
        "markdown": str(md_path),
    }

    def value(row: dict[str, Any], name: str) -> str:
        item = row.get(name, "N/A")
        return f"{item:.6f}" if isinstance(item, float) else str(item)

    expected = plan.get("expected", {})
    lines = [
        "# PSMA_MR dsba1 queued comparison",
        "",
        f"Status: {'terminal' if snapshot['terminal'] else 'in progress'} ({snapshot['timestamp']})",
        "",
        "Conditions: {method}, seed {seed}, support shots {shots}, preprocess {preprocess}, query split {split}, anomaly maps OFF, weights OFF.".format(
            method=expected.get("method", "PSMA_MR"),
            seed=expected.get("seed", 0),
            shots=expected.get("shots", 16),
            preprocess=expected.get("preprocess", "no_mask_no_rotation"),
            split=expected.get("query_split", "test_public"),
        ),
        "Results use native seg_AUPRO_0.05 and seg_AUROC_0.05 semantics. The current manifest schema records query count and fingerprints rather than query_paths; the collector requires those fingerprints and the configured test_public contract. No baseline value is invented or rerun.",
        "The v2 identity gate requires the resolved config, object metadata, output root, and no-mask/no-rotation context to match; retry2/v1 artifacts are not accepted.",
        "",
        "| Class | GPU | Queue | Result | seg_AUPRO_0.05 | seg_AUROC_0.05 | Fit s | Inference s | Conditions | Note |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {class_name} | {gpu} | {queue_status} | {result_status} | {aupro} | {auroc} | {fit} | {infer} | {conditions} | {note} |".format(
                class_name=row["class"],
                gpu=row["gpu"],
                queue_status=row["queue_status"],
                result_status=row["result_status"],
                aupro=value(row, "seg_AUPRO_0.05"),
                auroc=value(row, "seg_AUROC_0.05"),
                fit=row["fit_seconds"],
                infer=row["inference_seconds"],
                conditions=row["condition_status"],
                note=row["note"].replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            f"**Macro mean (full8 only):** seg_AUPRO_0.05={value(macro, 'seg_AUPRO_0.05')}, seg_AUROC_0.05={value(macro, 'seg_AUROC_0.05')}.",
            "",
            "Recovered artifacts are limited to metrics.json, config.yaml, and run_manifest.json under the local recovered/ directory.",
        ]
    )
    fd, temporary = tempfile.mkstemp(prefix=f".{md_path.name}.", dir=md_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, md_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def collector(plan_path: Path, interval: float, once: bool) -> int:
    plan = read_json(plan_path)
    if plan is None:
        print(f"invalid collector plan: {plan_path}", file=sys.stderr)
        return 2
    local_root = Path(str(plan["local_root"]))
    local_root.mkdir(parents=True, exist_ok=True)
    lock_path = local_root / ".collector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("collector already active (lock held)")
            return 0
        while True:
            snapshot = collect_once(plan)
            snapshot["collector_pid"] = os.getpid()
            snapshot["collector_starttime"] = local_process_starttime(os.getpid())
            snapshot["report_md"] = str(plan["report_md"])
            snapshot["report_csv"] = str(plan["report_csv"])
            canonical = append_canonical_ledger(plan, snapshot)
            snapshot["canonical_update"] = canonical
            write_reports(plan, snapshot)
            atomic_json(local_root / "collector_state.json", snapshot)
            if snapshot["terminal"]:
                print(
                    f"terminal report written: {plan['report_md']}; "
                    f"canonical ledger {canonical['status']} ({canonical['appended']} rows)"
                )
                return 0
            if once:
                print("collector checkpoint written; queue not terminal")
                return 0
            time.sleep(max(float(interval), 30.0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=45.0)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return collector(args.plan, args.interval, args.once)


if __name__ == "__main__":
    raise SystemExit(main())
