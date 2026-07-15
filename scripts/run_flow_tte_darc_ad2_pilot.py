# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
"""Generate frozen-support raw DARC ladder maps on MVTec AD2 public test images.

Run with ``FMAD_DINOV3_OFFLINE=1 python3 scripts/run_flow_tte_darc_ad2_pilot.py``
and the required data/output/object arguments. Add ``--good-limit 1 --bad-limit 1``
for the two-image execution smoke.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Optional, Sequence, Tuple, Union

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flow_tte.darc_ad2_pilot_io import TestLimits  # noqa: E402
from flow_tte.darc_ad2_pilot_runtime import (  # noqa: E402
    PilotRuntimeConfig,
    PreparedPilot,
    claim_fresh_output_root,
    pilot_query_id,
    prepare_pilot,
    run_pilot,
)
from flow_tte.darc_backbone import DINOv3EarlyExitAdapter  # noqa: E402
from flow_tte.darc_feature_stream import DarcFeatureStream, FeatureStreamConfig  # noqa: E402
from flow_tte.darc_gate2_provenance import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    REGISTERED_MODEL_PROVENANCE,
    file_sha256,
)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

JsonScalar = Union[None, bool, int, float, str]
JsonValue = Union[JsonScalar, Sequence["JsonValue"], Mapping[str, "JsonValue"]]

_DESIGN_PATH: Final = Path(
    "skill_graph/analysis/2026-07-11_flowtte_darc_ad2_raw_ladder_pilot.md",
)
_CODE_PATHS: Final = (
    Path("scripts/run_darc_common_eval.py"),
    Path("scripts/run_flow_tte_darc_ad2_pilot.py"),
    Path("src/flow_tte/darc_ad2_pilot.py"),
    Path("src/flow_tte/darc_ad2_pilot_io.py"),
    Path("src/flow_tte/darc_ad2_pilot_runtime.py"),
    Path("src/flow_tte/darc_backbone.py"),
    Path("src/flow_tte/darc_feature_stream.py"),
    Path("src/flow_tte/darc_common_eval.py"),
    Path("src/flow_tte/darc_common_report.py"),
    Path("src/flow_tte/darc_gate2_calibration.py"),
    Path("src/flow_tte/darc_gate2_coordinate_maps.py"),
    Path("src/flow_tte/darc_gate2_correspondence.py"),
    Path("src/flow_tte/darc_gate2_correspondence_types.py"),
    Path("src/flow_tte/darc_gate2_pipeline.py"),
    Path("src/flow_tte/darc_gate2_pipeline_audit.py"),
    Path("src/flow_tte/darc_gate2_pipeline_types.py"),
    Path("src/flow_tte/darc_gate2_provenance.py"),
    Path("src/flow_tte/darc_gate2_scoring.py"),
    Path("src/flow_tte/darc_gate2_scoring_types.py"),
    Path("src/flow_tte/darc_geometry.py"),
    Path("src/flow_tte/darc_knn.py"),
    Path("src/flow_tte/darc_map_io.py"),
    Path("src/flow_tte/darc_rank_metrics.py"),
    Path("src/flow_tte/darc_resources.py"),
    Path("src/flow_tte/darc_scoring.py"),
    Path("src/flow_tte/darc_tiling.py"),
    Path("src/flow_tte/metrics.py"),
    Path("src/flow_tte/superadd_morphology.py"),
)

_RUNTIME_PACKAGES: Final = (
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "Pillow",
    "scipy",
    "tifffile",
    "torch",
    "transformers",
    "typing_extensions",
)


# Python 3.8 dataclasses do not support slots=True.
@dataclass(frozen=True)
class _GridAdapter:
    adapter: DINOv3EarlyExitAdapter

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.adapter.extract(pixels).grids[0][0]


class _Args(argparse.Namespace):
    data_root: Path
    output_root: Path
    object: str
    device: str
    seed: int
    folds: str
    good_limit: Optional[int]
    bad_limit: Optional[int]
    query_chunk_size: int
    memory_chunk_size: int
    shard_index: int
    shard_count: int

    def __init__(self) -> None:
        super().__init__()
        self.data_root = Path()
        self.output_root = Path()
        self.object = ""
        self.device = "cuda:0"
        self.seed = 0
        self.folds = "0"
        self.good_limit = None
        self.bad_limit = None
        self.query_chunk_size = 256
        self.memory_chunk_size = 16384
        self.shard_index = 0
        self.shard_count = 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", default="0")
    parser.add_argument("--good-limit", type=int)
    parser.add_argument("--bad-limit", type=int)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--memory-chunk-size", type=int, default=16384)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def _parse_args() -> _Args:
    return _parser().parse_args(namespace=_Args())


def _parse_folds(text: str) -> Tuple[int, ...]:
    try:
        folds = tuple(int(part) for part in text.split(",") if part)
    except ValueError as error:
        raise ValueError("--folds must contain comma-separated integers") from error
    if not folds or len(set(folds)) != len(folds) or any(fold not in range(4) for fold in folds):
        raise ValueError("--folds must be a unique comma-separated subset of 0,1,2,3")
    return folds


def _load_stream(device: str) -> DarcFeatureStream:
    from transformers import AutoModel  # noqa: PLC0415

    local_only = os.environ.get("FMAD_DINOV3_OFFLINE", "").lower() in {"1", "true", "yes"}
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        local_files_only=local_only,
    )
    model = model.eval().to(device=device, dtype=torch.float32)
    micro = DINOv3EarlyExitAdapter(model, (7,), output_dtype=torch.float32)
    coarse = DINOv3EarlyExitAdapter(model, (23,), output_dtype=torch.float32)
    return DarcFeatureStream(
        micro_extractor=_GridAdapter(micro),
        coarse_extractor=_GridAdapter(coarse),
        config=FeatureStreamConfig(device=device, include_low=False),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(data_root: Path, path_text: str) -> str:
    return Path(path_text).relative_to(data_root).as_posix()


def _distribution_version(package: str) -> Optional[str]:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_environment() -> Mapping[str, JsonValue]:
    return {
        "python": platform.python_version(),
        "packages": {package: _distribution_version(package) for package in _RUNTIME_PACKAGES},
        "torch_cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
    }


def _write_manifest(config: PilotRuntimeConfig, prepared: PreparedPilot) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    selected_folds = [
        {
            "fold_index": fold.fold_index,
            "memory_paths": [_relative(config.data_root, path) for path in fold.memory_paths],
            "heldout_paths": [
                _relative(config.data_root, path) for path in fold.calibration_paths
            ],
        }
        for fold in prepared.folds
    ]
    payload: Mapping[str, JsonValue] = {
        "schema": "darc-ad2-raw-ladder-pilot-v1",
        "status": "performance-diagnostic-not-frozen-gate3",
        "data_root": str(config.data_root),
        "object": config.object_name,
        "seed": config.seed,
        "fold_indices": list(config.fold_indices),
        "source_pool_count": prepared.split.source_pool_count,
        "support_paths": [
            _relative(config.data_root, path) for path in prepared.split.support_paths
        ],
        "support_inventory": [
            {
                "path": _relative(config.data_root, path),
                "size": Path(path).stat().st_size,
                "sha256": file_sha256(Path(path)),
            }
            for path in prepared.split.support_paths
        ],
        "folds": selected_folds,
        "queries": [
            {
                "population": image.population.value,
                "path": image.path.relative_to(config.data_root).as_posix(),
                "size": image.path.stat().st_size,
                "scorer_query_id": pilot_query_id(config.object_name, image.path),
            }
            for image in prepared.test_images
        ],
        "test_limits": {"good": config.test_limits.good, "bad": config.test_limits.bad},
        "operational_shard": {
            "index": config.shard_index,
            "count": config.shard_count,
            "assignment": "global-good-then-bad-index-modulo",
        },
        "ground_truth_access_before_maps": False,
        "arms": ["G0", "L0", "L1", "R1"],
        "fold_aggregation": "float64-mean-then-single-float32-cast",
        "coverage_audit": (
            "per-fold pre-stitch scorer-token fallback and support-validity population"
        ),
        "model": REGISTERED_MODEL_PROVENANCE.to_manifest(),
        "runtime_environment": _runtime_environment(),
        "design": {"path": str(_DESIGN_PATH), "sha256": _sha256(ROOT / _DESIGN_PATH)},
        "code_sha256": {
            str(path): _sha256(ROOT / path) for path in _CODE_PATHS
        },
    }
    path = config.output_root / "pilot_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    try:
        folds = _parse_folds(args.folds)
        limits = TestLimits(good=args.good_limit, bad=args.bad_limit)
        config = PilotRuntimeConfig(
            data_root=args.data_root,
            output_root=args.output_root,
            object_name=args.object,
            device=args.device,
            seed=args.seed,
            fold_indices=folds,
            test_limits=limits,
            query_chunk_size=args.query_chunk_size,
            memory_chunk_size=args.memory_chunk_size,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        claim_fresh_output_root(config.output_root)
    except ValueError as error:
        _parser().error(str(error))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(mode=True)
    prepared = prepare_pilot(config)
    _write_manifest(config, prepared)
    print(
        json.dumps(
            {
                "status": "selection_frozen",
                "object": config.object_name,
                "support_count": len(prepared.split.support_paths),
                "fold_count": len(prepared.folds),
                "query_count": len(prepared.test_images),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    stream = _load_stream(args.device)
    report = run_pilot(config, prepared, stream)
    coverage_path = config.output_root / "coverage_rows.jsonl"
    with coverage_path.open("r", encoding="utf-8") as handle:
        persisted_coverage_rows = sum(1 for line in handle if line.strip())
    if persisted_coverage_rows != report.coverage_row_count:
        message = (
            "persisted coverage row count does not match the completed invocation: "
            f"{persisted_coverage_rows} != {report.coverage_row_count}"
        )
        raise RuntimeError(message)
    completed: Mapping[str, JsonValue] = {
        "schema": "darc-ad2-raw-ladder-pilot-completion-v1",
        "object": report.object_name,
        "seed": report.seed,
        "fold_count": report.fold_count,
        "image_count": report.image_count,
        "coverage_row_count": persisted_coverage_rows,
        "coverage_sha256": _sha256(coverage_path),
        "manifest_sha256": _sha256(config.output_root / "pilot_manifest.json"),
    }
    (config.output_root / "complete.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completed, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
