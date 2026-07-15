# /// script
# requires-python = ">=3.8"
# dependencies = []
# ///
# ─── How to run ───
# python3 scripts/run_flow_tte_darc_gate1.py --object bottle --seeds 0 --smoke
# python3 scripts/run_flow_tte_darc_gate1.py --aggregate

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flow_tte.darc_backbone import DINOv3EarlyExitAdapter  # noqa: E402
from flow_tte.darc_feature_stream import (  # noqa: E402
    DarcFeatureStream,
    FeatureStreamConfig,
)
from flow_tte.darc_gate1 import (  # noqa: E402
    Gate1Thresholds,
    decide_gate1,
)
from flow_tte.darc_gate1_aggregate import load_full_gate1_metrics  # noqa: E402
from flow_tte.darc_gate1_artifacts import gate1_method_manifest  # noqa: E402
from flow_tte.darc_gate1_provenance import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    code_bundle_hash,
)
from flow_tte.darc_gate1_runtime import (  # noqa: E402
    Gate1RuntimeConfig,
    prepare_gate1_run,
    run_gate1_seed,
)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

OBJECTS: Final = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
)


@dataclass(frozen=True)  # noqa: SLOTS_OK -- project supports Python 3.8
class _GridAdapter:
    adapter: DINOv3EarlyExitAdapter

    def __call__(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.adapter.extract(pixels).grids[0][0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the registered streaming DARC Gate1.")
    parser.add_argument("--data-root", type=Path, default=Path("/home/hunim/Volume/DATA/MVTecAD"))
    parser.add_argument("--output-root", type=Path, default=Path("results/darc_gate1"))
    parser.add_argument("--object", choices=OBJECTS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    return parser


def _parse_seeds(text: str) -> Tuple[int, ...]:
    seeds = tuple(int(part) for part in text.split(",") if part)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed not in (0, 1, 2) for seed in seeds):
        message = "--seeds must be a unique comma-separated subset of 0,1,2"
        raise ValueError(message)
    return seeds


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
        config=FeatureStreamConfig(device=device),
    )


def _aggregate(root: Path) -> int:
    aggregate = load_full_gate1_metrics(root, OBJECTS, (0, 1, 2))
    decision = decide_gate1(aggregate.metrics, Gate1Thresholds())
    payload = {
        "method": "darc-gate1-v1",
        "method_sha256": aggregate.method_sha256,
        "code_config_sha256": aggregate.code_config_sha256,
        **decision.to_manifest(),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "gate1_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (root / "gate_decision.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


def main() -> int:
    args = _parser().parse_args()
    if args.aggregate:
        return _aggregate(args.output_root)
    if args.object is None:
        _parser().error("--object is required unless --aggregate is used")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(mode=True)
    config = Gate1RuntimeConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        object_name=args.object,
        device=args.device,
        seeds=_parse_seeds(args.seeds),
        code_config_sha256=code_bundle_hash(ROOT, gate1_method_manifest()),
        smoke=args.smoke,
    )
    prepared = prepare_gate1_run(config)
    if not prepared.pending:
        print(json.dumps({"object": args.object, "status": "already_complete"}))
        return 0
    stream = _load_stream(args.device)
    for seed_run in prepared.pending:
        report = run_gate1_seed(config, seed_run, stream)
        progress = {
            "object": args.object,
            "seed": seed_run.split.seed,
            "sources": report.source_count,
        }
        print(json.dumps(progress))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
