# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
"""Generate frozen-support fine-branch density maps on MVTec AD2 public test images.

Emits three same-memory arms per query (``G0`` raw cosine residual, ``D0`` learned
density NLL, ``D0c`` normal-only calibrated density evidence) so the preregistered
fine density head can be compared against the raw cosine residual on identical
folds and tokens. Run with
``FMAD_DINOV3_OFFLINE=1 python3 scripts/run_flow_tte_darc_density_cell.py`` plus the
required data/output/object arguments. Add ``--good-limit 1 --bad-limit 1`` for the
two-image execution smoke.
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Final, Mapping, Optional, Sequence, Tuple, Union

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flow_tte.config import FlowConfig  # noqa: E402
from flow_tte.darc_ad2_density_runtime import (  # noqa: E402
    DensityRuntimeConfig,
    PreparedDensity,
    claim_fresh_output_root,
    prepare_density,
    run_density_cell,
)
from flow_tte.darc_ad2_pilot_io import TestLimits  # noqa: E402
from flow_tte.darc_backbone import DINOv3EarlyExitAdapter  # noqa: E402
from flow_tte.darc_feature_stream import DarcFeatureStream, FeatureStreamConfig  # noqa: E402
from flow_tte.darc_gate2_provenance import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    REGISTERED_MODEL_PROVENANCE,
)
from flow_tte.hres_density import HresDensityConfig  # noqa: E402

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

JsonScalar = Union[None, bool, int, float, str]
JsonValue = Union[JsonScalar, Sequence["JsonValue"], Mapping[str, "JsonValue"]]

_DESIGN_PATH: Final = Path(
    "skill_graph/analysis/2026-07-11_flowtte_hres_density_fusion_preregistration.md",
)
_CODE_PATHS: Final = (
    Path("scripts/run_flow_tte_darc_density_cell.py"),
    Path("src/flow_tte/hres_density.py"),
    Path("src/flow_tte/darc_ad2_density_runtime.py"),
    Path("src/flow_tte/darc_ad2_pilot_io.py"),
    Path("src/flow_tte/darc_backbone.py"),
    Path("src/flow_tte/darc_feature_stream.py"),
    Path("src/flow_tte/darc_knn.py"),
    Path("src/flow_tte/darc_resources.py"),
    Path("src/flow_tte/darc_scoring.py"),
    Path("src/flow_tte/darc_tiling.py"),
    Path("src/flow_tte/flow.py"),
    Path("src/flow_tte/trainer.py"),
    Path("src/flow_tte/losses.py"),
    Path("src/flow_tte/tensors.py"),
    Path("src/flow_tte/config.py"),
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


class _GridAdapter:
    def __init__(self, adapter: DINOv3EarlyExitAdapter) -> None:
        self.adapter = adapter

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
    n_coupling_layers: int
    n_epochs: int
    lr: float
    train_sample_cap: int
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
        self.n_coupling_layers = 4
        self.n_epochs = 30
        self.lr = 2e-4
        self.train_sample_cap = 131072
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
    parser.add_argument("--n-coupling-layers", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train-sample-cap", type=int, default=131072)
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
    import importlib.metadata  # noqa: PLC0415

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


def _write_manifest(config: DensityRuntimeConfig, prepared: PreparedDensity) -> None:
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
        "schema": "darc-ad2-density-cell-v1",
        "status": "fine-branch-shadow-diagnostic-not-frozen-gate3",
        "data_root": str(config.data_root),
        "object": config.object_name,
        "seed": config.seed,
        "fold_indices": list(config.fold_indices),
        "support_paths": [
            _relative(config.data_root, path) for path in prepared.split.support_paths
        ],
        "folds": selected_folds,
        "queries": [
            {
                "population": image.population.value,
                "path": image.path.relative_to(config.data_root).as_posix(),
            }
            for image in prepared.test_images
        ],
        "density": {
            "n_coupling_layers": config.density.flow.n_coupling_layers,
            "hidden_multiplier": config.density.flow.hidden_multiplier,
            "clamp": config.density.flow.clamp,
            "n_epochs": config.density.flow.n_epochs,
            "lr": config.density.flow.lr,
            "batch_size": config.density.flow.batch_size,
            "lambda_logdet": config.density.flow.lambda_logdet,
            "density_quantile": config.density.density_quantile,
            "train_sample_cap": config.density.train_sample_cap,
        },
        "test_limits": {"good": config.test_limits.good, "bad": config.test_limits.bad},
        "operational_shard": {
            "index": config.shard_index,
            "count": config.shard_count,
            "assignment": "global-good-then-bad-index-modulo",
        },
        "ground_truth_access_before_maps": False,
        "arms": ["G0", "D0", "D0c"],
        "fold_aggregation": "float64-mean-then-single-float32-cast",
        "model": REGISTERED_MODEL_PROVENANCE.to_manifest(),
        "runtime_environment": _runtime_environment(),
        "design": {"path": str(_DESIGN_PATH), "sha256": _sha256(ROOT / _DESIGN_PATH)},
        "code_sha256": {str(path): _sha256(ROOT / path) for path in _CODE_PATHS},
    }
    path = config.output_root / "density_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _parse_args()
    try:
        folds = _parse_folds(args.folds)
        limits = TestLimits(good=args.good_limit, bad=args.bad_limit)
        flow = FlowConfig(
            n_coupling_layers=args.n_coupling_layers,
            n_epochs=args.n_epochs,
            lr=args.lr,
            seed=args.seed,
        )
        density = HresDensityConfig(flow=flow, train_sample_cap=args.train_sample_cap)
        config = DensityRuntimeConfig(
            data_root=args.data_root,
            output_root=args.output_root,
            object_name=args.object,
            device=args.device,
            seed=args.seed,
            fold_indices=folds,
            density=density,
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
    prepared = prepare_density(config)
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
    report = run_density_cell(config, prepared, stream)
    completed: Mapping[str, JsonValue] = {
        "schema": "darc-ad2-density-cell-completion-v1",
        "object": report.object_name,
        "seed": report.seed,
        "fold_count": report.fold_count,
        "image_count": report.image_count,
        "manifest_sha256": _sha256(config.output_root / "density_manifest.json"),
    }
    (config.output_root / "complete.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(completed, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
