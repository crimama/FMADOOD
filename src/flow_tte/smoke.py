from __future__ import annotations

import json

import numpy as np
import numpy.typing as npt

from flow_tte.config import FlowTTEConfig
from flow_tte.evaluation import EvaluationBatch, EvaluationConfig, evaluate_flow_tte

FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


def main() -> None:
    rng = np.random.default_rng(7)
    support: FloatArray = rng.normal(loc=0.0, scale=0.5, size=(2, 8, 8, 8)).astype(
        np.float32,
    )
    batch: FloatArray = rng.normal(loc=0.1, scale=0.6, size=(2, 8, 8, 8)).astype(
        np.float32,
    )
    batch[1, 2:4, 2:4, :] = rng.normal(loc=4.0, scale=0.3, size=(2, 2, 8)).astype(np.float32)

    labels: BoolArray = np.array([0, 1], dtype=np.bool_)
    masks: BoolArray = np.zeros((2, 8, 8), dtype=np.bool_)
    masks[1, 2:4, 2:4] = True
    result = evaluate_flow_tte(
        support_features=support,
        batches=(EvaluationBatch(features=batch, image_labels=labels, pixel_masks=masks),),
        config=EvaluationConfig(
            pipeline_config=FlowTTEConfig.for_quick_probe(),
            expand=True,
        ),
    )
    payload = {
        "metrics": result.metrics.as_tte_dict(),
        "image_score": float(np.max(result.image_scores)),
        "image_scores": result.image_scores.tolist(),
        "patch_score_shape": (
            list(result.pixel_scores.shape) if result.pixel_scores is not None else None
        ),
        "memory_size_before": result.memory_sizes_before[0],
        "memory_size_after": result.memory_sizes_after[-1],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
