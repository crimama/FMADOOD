"""Durable, collision-safe result persistence for FMAD runs."""

import csv
import hashlib
import json
import os
import tempfile
import uuid
import copy
import fcntl
import importlib.metadata
import platform
import sys
import time
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import numpy as np

from fmad.methods.base import BaseMethod, ObjectResult


def validate_output_options(config: dict) -> dict:
    output = config.get("output", {})
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping")
    for key in ("save_anomaly_maps", "save_weights"):
        if not isinstance(output.get(key, False), bool):
            raise ValueError(f"output.{key} must be a boolean")
    return {key: output.get(key, False) for key in ("save_anomaly_maps", "save_weights")}


def _component(value: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"invalid artifact path component: {value!r}")
    return value


_TSV_COLUMNS = [
    "timestamp",
    "method",
    "dataset",
    "backbone",
    "seed",
    "object",
    "seg_AUROC_0.05",
    "seg_F1_oracle",
    "seg_AUPRO_0.05",
    "best_thre",
    "mean_inference_time",
    "memorybank_time",
    "metric_contract",
    "config_hash",
    "run_id",
]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _number(value: Any) -> str:
    if value is None:
        return ""
    return repr(value) if isinstance(value, (int, float)) else str(value)


def _compact(value: Any, *, debug: bool) -> Any:
    if debug:
        return value
    if isinstance(value, dict):
        return {
            k: _compact(v, debug=debug)
            for k, v in value.items()
            if debug or str(k).lower() != "prediction_diagnostics"
        }
    if isinstance(value, list):
        return [_compact(item, debug=debug) for item in value]
    return value


def _code_sha256() -> str:
    digest = hashlib.sha256()
    package_root = Path(__file__).parent
    for path in sorted(package_root.rglob("*.py")):
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "fsad-tta",
        "numpy",
        "scipy",
        "torch",
        "torchvision",
        "faiss-cpu",
        "faiss-gpu",
        "faiss-gpu-cu11",
        "faiss-gpu-cu12",
        "opencv-python",
        "opencv-python-headless",
        "scikit-learn",
        "Pillow",
        "PyYAML",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


class ResultsManager:
    """Manage isolated per-run outputs and an accumulated summary TSV."""

    def __init__(
        self,
        base_dir: str,
        method: str,
        dataset: str,
        backbone: str,
        seed: int,
        *,
        config: dict | None = None,
    ):
        self.method, self.dataset, self.backbone, self.seed = method, dataset, backbone, seed
        self.config = copy.deepcopy(config if config is not None else {})
        self.output_options = validate_output_options(self.config)
        self.config_hash = hashlib.sha256(_canonical_json(self.config).encode()).hexdigest()
        self.run_id = uuid.uuid4().hex
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.code_sha256 = _code_sha256()
        self.environment = _environment()
        parent = os.path.join(base_dir, method, dataset, backbone, f"seed={seed}")
        os.makedirs(parent, exist_ok=True)
        self.run_dir = os.path.join(parent, f"run-{self.config_hash[:16]}-{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=False)
        self.summary_path = os.path.join(base_dir, "summary.tsv")

    def log_training(self, object_name: str, event: str, **fields) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "object": object_name,
            "event": event,
            **fields,
        }
        encoded = json.dumps(row, allow_nan=False, sort_keys=True)
        with open(Path(self.run_dir) / "training.jsonl", "a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")

    @contextmanager
    def training_session(self, method, object_name: str, support_count: int):
        previous = None
        if isinstance(method, BaseMethod):
            previous = method._training_log_sink
            method._training_log_sink = lambda step, metrics: self.log_training(
                object_name, "step", step=step, metrics=metrics
            )
        started = time.perf_counter()
        try:
            self.log_training(object_name, "fit_started", support_count=support_count)
            yield
        except Exception as error:
            self.log_training(
                object_name,
                "fit_failed",
                error_type=type(error).__name__,
                error=str(error),
                elapsed_seconds=time.perf_counter() - started,
            )
            raise
        else:
            self.log_training(
                object_name, "fit_finished", elapsed_seconds=time.perf_counter() - started
            )
        finally:
            if isinstance(method, BaseMethod):
                method._training_log_sink = previous

    def save_anomaly_map(self, object_name: str, sample_key: str, score_map: np.ndarray) -> None:
        if not self.output_options["save_anomaly_maps"]:
            return
        parts = sample_key.split("/")
        if len(parts) != 2:
            raise ValueError("sample key must be anomaly_type/filename")
        directory = (
            Path(self.run_dir) / "anomaly_maps" / _component(object_name) / _component(parts[0])
        )
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{_component(parts[1])}.npz"
        array = np.asarray(score_map, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("anomaly map must be a finite 2D array")
        fd, temporary = tempfile.mkstemp(prefix=".map-", dir=directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez_compressed(stream, score_map=array)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save_weights(self, method, object_name: str) -> str | None:
        if not self.output_options["save_weights"]:
            return None
        parent = Path(self.run_dir) / "weights"
        parent.mkdir(exist_ok=True)
        target = parent / _component(object_name)
        if target.exists():
            raise FileExistsError(target)
        staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=parent))
        try:
            method.save_checkpoint(staging)
            if not any(staging.iterdir()):
                raise ValueError("method produced an empty checkpoint")
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        relative = target.relative_to(self.run_dir).as_posix()
        self.log_training(object_name, "checkpoint_saved", path=relative)
        return relative

    def _atomic_json(self, name: str, payload: Any) -> None:
        fd, tmp = tempfile.mkstemp(prefix=f".{name}.", dir=self.run_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, os.path.join(self.run_dir, name))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def save_config(self, config: dict) -> None:
        import yaml

        if _canonical_json(config) != _canonical_json(self.config):
            raise ValueError("save_config must match the constructor configuration")
        fd, tmp = tempfile.mkstemp(prefix=".config.", dir=self.run_dir)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=True, default_flow_style=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, os.path.join(self.run_dir, "config.yaml"))
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def save_object_times(self, results: List[ObjectResult]) -> None:
        if not (
            self.config.get("debug_output", False)
            or self.config.get("output", {}).get("debug_output", False)
        ):
            return
        with open(
            os.path.join(self.run_dir, "time_measurements.csv"), "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["object", "sample", "anomaly_score", "memorybank_time", "inference_time"]
            )
            for result in results:
                for sample_key, score in result.anomaly_scores.items():
                    writer.writerow(
                        [
                            result.object_name,
                            sample_key,
                            _number(score),
                            _number(result.time_memorybank),
                            _number(result.inference_times[sample_key]),
                        ]
                    )

    def save_metrics(self, metrics: dict) -> None:
        self._atomic_json("metrics.json", metrics)

    def save_run_manifest(self, results: List[ObjectResult]) -> None:
        debug = bool(
            self.config.get("debug_output", False)
            or self.config.get("output", {}).get("debug_output", False)
        )
        self._atomic_json(
            "run_manifest.json",
            {
                "run_id": self.run_id,
                "config_hash": self.config_hash,
                "code_sha256": self.code_sha256,
                "environment": self.environment,
                "objects": {r.object_name: _compact(r.metadata, debug=debug) for r in results},
            },
        )

    def append_summary(self, metrics: dict, results: List[ObjectResult]) -> None:
        times = {
            r.object_name: {
                "mean": sum(r.inference_times.values()) / len(r.inference_times),
                "memorybank": r.time_memorybank,
            }
            for r in results
            if r.inference_times
        }
        objects = {r.object_name for r in results}
        os.makedirs(os.path.dirname(self.summary_path), exist_ok=True)
        with open(self.summary_path, "a+", newline="", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                stream.seek(0)
                existing_header = stream.readline().rstrip("\r\n")
                expected_header = "\t".join(_TSV_COLUMNS)
                if existing_header and existing_header != expected_header:
                    raise ValueError("summary.tsv header does not match the current contract")
                stream.seek(0, os.SEEK_END)
                writer = csv.DictWriter(stream, fieldnames=_TSV_COLUMNS, delimiter="\t")
                if stream.tell() == 0:
                    writer.writeheader()
                for name, values in metrics.items():
                    if not isinstance(values, dict) or name not in objects:
                        continue
                    timing = times.get(name, {})
                    writer.writerow(
                        {
                            "timestamp": self.timestamp,
                            "method": self.method,
                            "dataset": self.dataset,
                            "backbone": self.backbone,
                            "seed": self.seed,
                            "object": name,
                            "seg_AUROC_0.05": _number(values.get("seg_AUROC_0.05")),
                            "seg_F1_oracle": _number(values.get("seg_F1_oracle")),
                            "seg_AUPRO_0.05": _number(values.get("seg_AUPRO_0.05")),
                            "best_thre": _number(values.get("best_thre")),
                            "mean_inference_time": _number(timing.get("mean")),
                            "memorybank_time": _number(timing.get("memorybank")),
                            "metric_contract": values.get(
                                "metric_contract",
                                self.config.get("evaluation", {}).get("metric_contract", ""),
                            ),
                            "config_hash": self.config_hash,
                            "run_id": self.run_id,
                        }
                    )
                stream.flush()
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
