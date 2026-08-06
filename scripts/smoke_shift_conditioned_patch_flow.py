#!/usr/bin/env python3
"""Dependency-free runtime smoke for the shift-conditioned experiment stack."""

from __future__ import annotations

import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_shift_conditioned_patch_flow import (  # noqa: E402
    FeatureMap,
    PhotometricTransformSpec,
    StageBPair,
    TorchShiftConditionedBackend,
    frozen_config,
)

from flow_tte.flow import PatchNormalizingFlow  # noqa: E402
from flow_tte.shift_conditioned import ShiftConditionedPatchLocalFlow  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    base = PatchNormalizingFlow(
        dim=4,
        n_coupling_layers=2,
        hidden_multiplier=1,
        clamp=1.9,
    )
    model = ShiftConditionedPatchLocalFlow(base, patch_dim=4)
    patches = torch.randn(2, 4, 4)
    descriptor = model.encode_shift(patches).descriptor
    base_z, base_logdet = base(patches.reshape(-1, 4))
    initial = model(patches, descriptor=descriptor)
    if not torch.allclose(initial.z.reshape(-1, 4), base_z, atol=1e-6, rtol=0.0):
        raise RuntimeError("zero-head parity failed")
    if not torch.allclose(initial.logdet.reshape(-1), base_logdet, atol=1e-6, rtol=0.0):
        raise RuntimeError("zero-head logdet parity failed")
    with torch.no_grad():
        model.residual_heads[0].delta.weight.fill_(1e-3)
    forward = model(patches, descriptor=descriptor)
    reverse = model(forward.z, reverse=True, descriptor=descriptor)
    inverse_error = float((reverse.z - patches).abs().max().detach())
    logdet_error = float((forward.logdet + reverse.logdet).abs().max().detach())
    if inverse_error > 1e-5 or logdet_error > 1e-5:
        raise RuntimeError("nontrivial conditional inverse failed")

    rng = np.random.default_rng(0)
    support = tuple(
        FeatureMap(
            values=rng.normal(size=(2, 2, 4)).astype(np.float32),
            grid=(2, 2),
            image_shape=(32, 32),
            relative_path=f"fabric/train/good/{index:03d}.png",
        )
        for index in range(3)
    )
    pair = StageBPair(
        source=support[0],
        transformed=FeatureMap(
            values=(support[0].values + np.float32(0.05)).astype(np.float32),
            grid=(2, 2),
            image_shape=(32, 32),
            relative_path=support[0].relative_path,
            transform="brightness",
        ),
        transform=PhotometricTransformSpec(
            name="brightness",
            active_axes=("brightness",),
            brightness=1.1,
        ),
    )
    config = deepcopy(frozen_config("cpu"))
    config["stage_a_flow"]["n_epochs"] = 1
    config["stage_a_flow"]["batch_size"] = 32
    config["stage_b"]["epochs"] = 1
    backend = TorchShiftConditionedBackend()
    backend.set_dvt_artifact(None, alpha=1.0)
    backend.fit_stage_a(support, config)
    log = backend.train_stage_b((pair,), config)
    before = backend.score_maps(support, arm="E5", descriptor_mode="correct")
    with tempfile.TemporaryDirectory(prefix="shift-flow-smoke-") as directory:
        checkpoint = Path(directory) / "checkpoint.pt"
        backend.save_checkpoint(checkpoint)
        restored = TorchShiftConditionedBackend()
        restored.load_checkpoint(checkpoint, config)
        after = restored.score_maps(support, arm="E5", descriptor_mode="correct")
    roundtrip_error = max(
        float(np.max(np.abs(left - right)))
        for left, right in zip(before, after)
    )
    if roundtrip_error != 0.0:
        raise RuntimeError("checkpoint score roundtrip failed")
    print(
        json.dumps(
            {
                "status": "passed",
                "inverse_max_error": inverse_error,
                "logdet_max_error": logdet_error,
                "checkpoint_score_max_error": roundtrip_error,
                "stage_b_pair_count": log["pair_count"],
            },
            sort_keys=True,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
