# PSMA_MR method guide

PSMA_MR is a first runnable, support-only candidate in the common FMAD-OOD
pipeline. Its purpose is to test whether a small shared residual adapter can
reduce normal nuisance sensitivity while retaining synthetic defect evidence at
more than one input resolution. The result of this first run is directional
development evidence; this guide makes no SOTA, final-benchmark, or completed
test claim.

The implementation has an explicit `method_config.revision` switch. Existing
configs without that key continue to use the reproducible v1 path; the
approved dsba1 candidate is v2 and is selected with:

```bash
python run.py --config configs/methods/PSMA_MR_v2.yaml \
  --method PSMA_MR --data-root /data/mvtec_ad_2
```

## Fixed method boundary

The common runner owns dataset enumeration, seeded support selection, query
ordering, prediction validation, streaming metrics, and result writing. For an
object, PSMA_MR receives the selected normal support paths and immutable
`MethodContext` only. It must not receive the dataset object, query labels,
ground truth, or benchmark query images during fitting.

The method uses the frozen common pretrained backbone and one shared trainable
residual adapter on the final selected feature layer. The first candidate fixes
the adapter rank at 4. Training uses paired support-derived views:

- a normal-preserving nuisance pair, for score consistency; and
- a clean/synthetic-local-anomaly pair, for a defect-score margin.

A feature-anchor term keeps the adapted representation tied to the frozen
feature geometry. The exact synthetic renderer and optimizer schedule are
implementation details of this first candidate, not additional comparison
factors.

Reference construction is leave-source-out. For a source support image, that
image and every nuisance or synthetic descendant made from it are excluded from
the references used to train or score that source. This prevents a generated
view from matching its own source and keeps the training signal support-only.
The source image identities remain the same across resolutions.

At inference, each configured resolution builds/uses its own cosine nearest-
neighbour patch bank from the same support identities. Maps are resized to the
query's native image dimensions and averaged across resolutions. The fused
map's common top-1% statistic supplies the image score. Queries do not update
the adapter, banks, or any calibration state.

The immutable context flags are part of the method contract. `rotation=true`
creates the same eight fixed support rotations used by SuperAD, while queries
are scored without rotation augmentation. `masking=true` is passed to the
shared backbone foreground-mask operation for support selection and query
scoring; if the backbone lacks that operation, fitting raises an explicit
error. A synthetic family with no visible foreground token after masking may
be skipped and is recorded in training diagnostics, so tiny inputs are not
required to expose every family. Use `dataset_config.preprocess:
no_mask_no_rotation` when the neutral no-mask/no-rotation condition is the
intended control.

The common frozen backbone is loaded once and reused across all resolutions via
the non-mutating `prepare_image(image, resolution=...)` path. With DINOv2,
feature extraction follows the configured `--device`, but extracted features
are CPU arrays; PSMA_MR's adapter, support banks, and nearest-neighbour search
therefore run in CPU PyTorch even when the frozen backbone uses CUDA. `--device
cpu` runs the complete method on CPU.

## Configuration and controls

PSMA_MR-specific values are flattened under `method_config`:

```yaml
method: PSMA_MR
method_config:
  adapter_enabled: true       # bool; default true
  adapter_rank: 4              # default 4 for the first candidate
  resolutions: [448, 672]      # default; one value selects single-resolution
  steps: 24                    # adapter updates; 0 disables updates
  learning_rate: 0.001
```

Additional flat training controls are `weight_decay` (0.0), `margin` (0.10),
`normal_consistency_weight` (1.0), `synthetic_margin_weight` (1.0), and
`feature_anchor_weight` (0.10). Bounded-memory controls are
`max_patches_per_source` (256), `max_bank_patches` (same as the source cap),
`max_train_patches` (96), and `nn_chunk_size` (1024); view controls are
`photometric_strength` (1.0), `synthetic_anomaly_types`, and `gradient_clip`
(5.0). The canonical names are `steps` and `learning_rate`; `train_steps` and
`lr` aliases are not accepted.

The exact first-factor matrix is adapter on/off crossed with single/multi
resolution. For a single-resolution arm, use the same one-element list (for
example `[672]`) in both adapter arms. For a multi-resolution arm, use the
default `[448, 672]` in both adapter arms:

| Arm | `adapter_enabled` | `resolutions` | Interpretation |
| --- | --- | --- | --- |
| frozen/single | `false` | `[672]` | frozen one-resolution control |
| adapted/single | `true` | `[672]` | adapter contribution at one resolution |
| frozen/multi | `false` | `[448, 672]` | fusion contribution without adaptation |
| PSMA_MR | `true` | `[448, 672]` | full first candidate |

`adapter_rank` stays at 4 for all four arms. Do not introduce a nested
`method.params` compatibility block or silently change the common protocol.

## CLI workflow

The common source-checkout entry point selects the registered method by its exact
name:

```bash
python run.py --config configs/default.yaml --method PSMA_MR \
  --data-root /data/mvtec_ad_2 --device cuda:0 \
  --output-dir ./results
```

The integration's supported CLI overrides for the two scalar/list method values
are:

```bash
python run.py --config configs/default.yaml --method PSMA_MR \
  --resolutions 448,672 --adapter-rank 4 \
  --data-root /data/mvtec_ad_2 --device cuda:0
```

Use `method_config.adapter_enabled` in YAML for the adapter on/off arm. Common
framework controls remain available through their existing names, including
`--seed`, `--shots`, `--objects`, `--fixed-reference-manifest`,
`--debug-output`, `--save-anomaly-maps`, and `--save-weights`. There is no
generic `--method-config` or `--multi-resolution` option; PSMA_MR settings are
YAML `method_config` entries or the two explicit overrides above.

The runner loads YAML, applies supported CLI overrides, validates the protocol,
seeds before constructing the model, selects the support set, and runs the
shared fit/predict/evaluate loop for each object. Use the same seed, support
manifest or shot count, dataset, backbone, preprocessing, query population, and
evaluator when comparing the four arms. The current MVTec AD 2 runner's default
query population is `test_public`; without a separately verified sealed split,
record the actual label exposure and treat all numbers as development evidence.

## Outputs and first-result policy

The common output policy applies unchanged:

- `config.yaml`, `metrics.json`, `run_manifest.json`, and `training.jsonl` are
  the default run artifacts; `summary.tsv` records the object/timing rows.
- Streaming evaluation quantizes scores to float16 for bounded-memory metric
  aggregation. This is a storage/evaluation contract, not a claim about model
  precision.
- Anomaly maps and fitted adapter checkpoints are opt-in and off by default:
  set `output.save_anomaly_maps` or `output.save_weights` to `true`, or use the
  corresponding supported CLI output flags. `--debug-output` is independent
  and adds timing/diagnostic output.

No remote experiment is implied by this guide. Before reporting a first result,
verify run completion and preserve the resolved config, support/query custody,
training log, and manifest. Report the four controls and common metrics under
the same conditions; describe any interrupted or exposed run as incomplete or
development-only rather than as a method failure or a final claim.

## Approved v2 contract

`configs/methods/PSMA_MR_v2.yaml` is the explicit dsba1 configuration. It fixes
the DINOv2 ViT-L/14 outputs to code layers `[5, 11, 17, 23]` at short edges
`448` and `672`. All four backbone branches are frozen; only final layer 23
uses one shared rank-4 residual. The residual is
`h = U*tanh(V*LN(z))`, with `U=0`, seeded-random `V`, and a zero-safe clip of
`||h||` to `0.1*||z||` before `normalize(z+h)`. This retains a nonzero initial
gradient for `U` whenever the normalized feature is informative and trains no
all-layer LoRA or other backbone parameters.

Support caches retain descendant/source IDs and original grid coordinates
through masking and rotation. Selection is deterministic spatial stratification
per source and branch, with caps of 256 final tokens and 128 intermediate
tokens; final training samples use 96 tokens while preserving synthetic mask
positives. Synthetic identity/nuisance caches are final-layer-only. When the
backbone returns all four outputs, one extraction pass is reused rather than
running once per layer. Leave-one-source-out banks exclude every descendant of
the held-out source; an empty foreground follows the existing masking contract
(explicit masking errors, neutral no-mask falls back to all tokens).

For each layer/resolution branch, raw cosine nearest-neighbour distances are
calibrated once after the adapter is final using clean normal support tokens and
source-excluded banks only. Calibration is `(d-median)/max(1.4826*MAD, .001)`
with no upper or lower clipping, no query images, and no ground-truth labels.
With 16 support images this is LOO15 calibration followed by an all-support16
inference bank; adapter fitting itself uses all 16 support images and is marked
as a non-strict holdout. The eight calibrated maps are resized to native query
size and equally averaged; the image score remains the fused map's mean top 1%.
Adapter-off v2 constructs its own frozen banks and calibration under the same
contract and is not a historical SuperAD-equivalence claim.

The default v2 schedule is `t=0..31`,
`source=floor(t/2)%16`, `resolution=res[t%2]`, with Adam `lr=.001`,
`weight_decay=0`, gradient clip `5`, and consistency/local/anchor weights
`1/1/.1`. The clean and synthetic identity/nuisance hinge is applied per
aligned token before averaging: `mean(relu(.1 + d_clean - d_synthetic))`.
The config explicitly uses `no_mask_no_rotation`, seed `0`, 16 shots, and keeps
anomaly-map/checkpoint outputs disabled.
