# AD2 Random 16-Shot DINOv2-L/14 — Preregistration

Date: 2026-07-13

## Negative Evidence Intake and motivation

The full-normal experiment did not improve over the fixed 16-shot proposed
setting, suggesting that support volume is not the main AD2 bottleneck. This
experiment measures the simpler random few-shot formulation with the
paper-aligned DINOv2 ViT-L/14 backbone. It is a data-efficiency diagnostic,
not a new mechanism and not a retune of a previously killed score component.

## Frozen experiment contract

- `target_dataset=MVTec AD2 single-image`
- `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`
- Objects: all eight public categories and full `test_public/good,bad`.
- Support: exactly 16 paths sampled without replacement from each sorted
  `train/good` pool by NumPy `default_rng(0)`; policy
  `visionad_seeded_random`, selection seed 0.
- Backbone: DINOv2 ViT-L/14, code layers `[5,11,17,23]`, shorter-side 672
  except `sheet_metal=448`.
- Unchanged proposed components: Flow 3 epochs, 2 couplings, hidden
  multiplier 1, static latent bank, latent 1-NN with LOO standardization,
  DVT position-mean alpha 1, density weight 0.25, RGB guided-r8, and fixed
  line17/angles16 close/fill/erode.
- No TTE expansion, support transform, brightness augmentation, context,
  register routing, or tiling.
- Metrics: class-macro and per-object `seg_AUROC_0.05`, raw best-threshold
  pixel F1, and post-morphology pixel F1.
- The intermediate unguided map export uses `binary_postprocess=none`; this
  avoids a redundant morphology pass. The result-bearing guided evaluator
  still applies the frozen line17/angles16 close/fill/erode profile.

## Evaluation alignment and gates

The recorded DINOv3-H+ fixed-support proposed result and full-normal result
are context only because both the backbone and support policy differ. A
same-condition random-seed DINOv2 SuperAD/RN-FMLK baseline is absent, so the
strict claim verdict is capped at `BLOCKED_BASELINE`.

This single-seed run supports a continuation only if all eight rows are
finite, exact random support manifests are auditable, memories remain static,
and the macro metrics retain a plausible fraction of the existing proposed
result. It cannot support a variance or random-support robustness claim; that
requires a separately preregistered multi-seed run.

## Remote and cleanup contract

Run four independent two-object shards on dsba3 GPUs 0--3 in the fixed
project container. Pull compact metrics, manifests, support paths, and logs;
delete regenerable anomaly maps after guided evaluation and verify cleanup.

## Pre-result execution amendments

- `v1` and `v2` stopped before feature extraction because the newly installed
  transformers package expected a newer public PyTorch PyTree API. They
  produced no metric.
- `v3` verified the compatibility shim and reached test scoring, but was
  stopped before metric emission after detecting a redundant unguided
  morphology pass in the inherited launcher.
- `v4` did not start with the updated launcher because the remote volume was
  full; it produced no metric. `v5` verified the optimized command but was
  stopped before metrics when eight simultaneous dense-map trees exhausted
  the remaining volume.
- `v6_first4` showed that four raw map trees fit, but four additional guided
  copies would exceed the volume; it was stopped before metrics.
- `v7_first4` and `v7_last4` are the sole result-bearing waves. Each runs one
  object per GPU and refines its map files in place before evaluation and
  cleanup, while preserving separate raw and guided compact manifests. They
  disable only the intermediate binary postprocess; final guided-r8
  morphology remains unchanged.
