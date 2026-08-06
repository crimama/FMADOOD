# Baseline1 Method Specification

Date named: 2026-08-05

## Identity

`Baseline1` is the canonical name for the method previously described as the
strongest retained/default FlowTTE configuration. It is a static, patch-local
Flow latent-memory detector: no Transformer token mixing, register/FiLM
conditioning, query adaptation, or test-time memory update is used.

Completed paired-loss v4 artifacts use the historical machine ID
`V0_default`. That ID remains immutable for checksum compatibility and is an
alias of `Baseline1`, not another method.

## End-to-end structure

```text
normal support-16
  -> frozen DINOv3-H+/16 patch features [7,15,23,31]
  -> per-layer normalization + layer mean fusion
  -> DVT position-mean denoising (alpha 1.0)
  -> standardized patch-local MLP affine-coupling Flow
  -> fixed support latent memory

query image
  -> identical frozen feature/DVT/Flow path
  -> raw latent 1-NN distance to support memory
  -> support-LOO score standardization
  -> optional frozen density term used by the full Baseline1 scorer
  -> anomaly map
```

## Frozen configuration

| Component | Baseline1 setting |
|---|---|
| Target | MVTec AD2, single image |
| Objects | all 8 public objects for the recorded anchor |
| Support | fixed SuperAD-selected 16 normal images/object, seed 0 |
| Support manifest SHA-256 | `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5` |
| Support view | identity; one deterministic brightness factor in `[0.8,1.2]` |
| Backbone | frozen `dinov3_vith16plus` (DINOv3-H+/16) |
| Layers | `[7,15,23,31]` |
| Resolution | 672; `sheet_metal` 448 under the dataset-default contract |
| Feature fusion | `layer_norm_mean` |
| DVT | `position_mean`, alpha `1.0` |
| Flow topology | patch-local MLP affine coupling; no spatial/context mixing |
| Coupling layers | 2 |
| Hidden multiplier | 1 |
| Training | 3 epochs, LR `2e-4`, batch size 512, seed 0 |
| Affine clamp | `1.9` |
| Feature standardization | enabled |
| Tail loss | weight `0.3`, top-k ratio `0.05` |
| Logdet regularization | `lambda_logdet=0.02` |
| Memory | fixed support latent bank; expansion budget `1.0`; no TTE |
| Main score | raw Euclidean latent 1-NN, distance weight `1.0` |
| Score calibration | support LOO standardization; sample size 4096 |
| Full scorer density | weight `0.25`, support quantile `0.90` |
| Context/register | none |
| Query adaptation | none |
| Tiling/masking/rotation | none |

## Recorded anchor

The all-eight Phase-3 tuned diagnostic recorded:

| Metric | Value |
|---|---:|
| `seg_AUROC_0.05` | `0.837426` |
| oracle pixel `seg_F1` | `0.530635` |

These values are development/shadow context, not a new same-condition
SuperAD superiority claim.

## Baseline1 versus Baseline1-core

The 2026-08-04 paired-loss mechanism experiment intentionally set
`density_weight=0.0` so that only raw latent geometry was tested. Its
`V0_default` row should therefore be read as `Baseline1-core`:

- identical backbone, layers, DVT, MLP Flow, support, and latent memory;
- density/NLL contribution removed from the final anomaly score;
- support-only mechanism metrics, with no TESTpublic access.

Unless an experiment explicitly says `Baseline1-core`, future full benchmark
comparisons should use the complete Baseline1 scorer with density weight
`0.25` and state any postprocessing as a separate `Baseline1+Post` arm.

## Explicit exclusions

- all-token Transformer Flow;
- task register or FiLM residual conditioner;
- CLS/register support routing;
- query/test-time optimization or memory expansion;
- class-specific tuning;
- RGB guided filtering or morphology as an implicit part of the core method.

RGB guidance and morphology, when evaluated, must be named separately as an
output-chain arm rather than silently included in Baseline1.
