# FlowTTE integration scaffold

This workspace now contains a small integration package for testing the proposed
normalizing-flow variant of TTE.

The package keeps the TTE contract:

```text
support image patch embeddings -> fit M0 memory + fit flow density
test image patch embeddings -> score with calibrated latent memory -> select by flow density -> expand reservoir
```

The DeCoFlow-inspired part is intentionally limited to the density-learning
pieces needed for this project direction:

- affine-coupling normalizing flow over patch embeddings
- patch NLL from `log p(z) + log|det J|`
- per-image tail-aware loss for hard normal patches
- `logdet^2` regularization to avoid uncontrolled volume expansion
- M0 leave-one-out distance calibration so latent distance and NLL penalties are comparable
- chunked kNN memory queries to keep large patch banks bounded in memory

It does not include continual-learning adapters, routing, or task isolation.

## Smoke run

```bash
PYTHONPATH=src python3 -m flow_tte.smoke
```

## Python API

```python
import numpy as np

from flow_tte import FlowTTE, FlowTTEConfig

support = np.random.normal(size=(2, 16, 16, 384)).astype("float32")
batch = np.random.normal(size=(1, 16, 16, 384)).astype("float32")

pipeline = FlowTTE(FlowTTEConfig.for_quick_probe())
pipeline.fit(support)
result = pipeline.score_then_expand(batch)

print(result.patch_scores.shape, result.image_scores, result.memory_size_after)
```

## Evaluation loop

`evaluate_flow_tte` runs the full score-expand-evaluate loop and returns the same
metric set used by the original TTE evaluation:

```python
import numpy as np

from flow_tte import EvaluationBatch, EvaluationConfig, FlowTTEConfig, evaluate_flow_tte

support = np.random.normal(size=(2, 16, 16, 384)).astype("float32")
test = np.random.normal(size=(4, 16, 16, 384)).astype("float32")
image_labels = np.array([0, 1, 0, 1], dtype=np.bool_)
pixel_masks = np.zeros((4, 16, 16), dtype=np.bool_)

result = evaluate_flow_tte(
    support_features=support,
    batches=(EvaluationBatch(test, image_labels=image_labels, pixel_masks=pixel_masks),),
    config=EvaluationConfig(
        pipeline_config=FlowTTEConfig.for_quick_probe(),
        expand=True,
    ),
)

print(result.metrics.as_tte_dict())
```

The metric keys are `I-AUROC`, `I-AP`, `I-F1_max`, `P-AUROC`, `P-AP`,
`P-F1_max`, and `AUPRO`. Pixel masks are optional, but if provided they must be
provided for every batch and match the returned patch-score map shape.

`FlowTTE` accepts:

- `(n_patches, dim)` for flat patch probes, returning flat `(n_patches,)` scores.
- `(n_images, patches, dim)` for image-grouped patch probes.
- `(n_images, height, width, dim)` for localization, returning `(n_images, height, width)` maps.

## How to plug into the original TTE repo

Use the DINO patch embeddings currently passed into TTE scoring/expansion:

1. Fit `FlowTTE` on support embeddings before flattening away image/spatial shape.
2. Replace raw memory storage with `pipeline.transform_features(...)` if the host
   pipeline needs latent features directly.
3. During online test, call `score_then_expand(batch_feats)` before absorbing the
   same batch into memory.

This makes expansion operate in flow-normalized latent space while the density
gate and distance calibration remain anchored to the host `M0` flow.
