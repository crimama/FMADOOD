# FlowTTE Paired-Loss Diagnostic Preregistration

## Motivation and boundary

The previous shift-conditioned register experiment learned the photometric
shift probe but worsened held shifted-normal raw 1-NN inflation by 20.41% and
failed clean identity and local-defect retention.  This run removes the
register, FiLM, and Transformer variables and asks one question only:

> Can a loss aligned to the raw latent support-memory distance canonicalize
> held photometric normal shifts without moving clean support geometry or
> attenuating local defect evidence?

This is not a retune of the killed register branch.  It is a support-only loss
diagnostic on the retained patch-local MLP Flow.  No query adaptation, TTE,
test split, ground truth, or public metric is permitted in Stage -1.

## Frozen target and Baseline1

The strongest method previously called the retained/default MLP Flow is named
`Baseline1` from 2026-08-05 onward. Completed artifacts retain the historical
machine arm ID `V0_default`; this is an identifier alias, not a different
method.

- `target_dataset=MVTec AD2 single-image`
- `data_root=/home/hunim/Volume/DATA/mvtec_ad_2`
- Stage -1 objects: `fruit_jelly, walnuts, vial, fabric`
- exact fixed SuperAD-selected 16 support paths; manifest SHA-256
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- DINOv3-H+/16, layers `[7,15,23,31]`, `layer_norm_mean`, native 672
- DVT `position_mean`, alpha `1.0`
- patch-local MLP Flow: 2 coupling layers, hidden multiplier 1, 3 epochs,
  learning rate `2e-4`, clamp `1.9`, tail weight `0.3`, tail top-k `0.05`
- strongest retained loss setting: `lambda_logdet=2e-2`
- strongest retained deterministic support preprocessing: one brightness
  factor per selected support, sampled in `[0.8,1.2]` with seed 0
- mechanism scorer: raw Euclidean latent 1-NN, support-LOO standardization,
  `density_weight=0.0`
- recorded all-eight public context for this baseline configuration:
  `seg_AUROC_0.05=0.837426`, `seg_F1=0.530635`; it is context only and is not
  reused as a Stage -1 gate.

The same-condition SuperAD/RN-FMLK public comparison is not invoked in this
support-only stage.  Any later public promotion remains blocked until this
mechanism gate passes.

## Paired data

Each brightness-preprocessed support image is the canonical source.  Training
and held transforms use disjoint deterministic seeds.  The photometric axes
are brightness, contrast, gamma, RGB gain, Gaussian blur, and Gaussian noise,
with the frozen 20% identity / 50% one-axis / 30% two-axis mixture.  Patch
correspondence is exact.

The Stage-A teacher `F0` is fit once.  Every trainable arm starts from the same
`F0` weights and standardizer.  Canonical targets are detached
`z0 = stopgrad(F0(x_clean))`.

## Frozen arms

| Arm | Stage-B objective | Purpose |
|---|---|---|
| `Baseline1` (`V0_default` in v4 artifacts) | no Stage-B update | strongest MLP control |
| `V1_aug_nll` | clean + augmented NLL/logdet | augmentation-exposure control |
| `V2_raw_pair_nll1` | raw pair + clean identity + norm + NLL weight 1.0 | metric-aligned candidate |
| `V3_raw_pair_nll01` | same, NLL weight 0.1 | test NLL conflict |
| `C1_normalized_pair` | normalized Huber pair + identity + NLL 0.1 | prior geometry control |
| `C2_random_pair` | raw pair to a wrong clean target + identical regularizers | collapse/non-causal null |

Stage B uses 5 epochs, learning rate `1e-4`, weight decay `1e-4`, gradient
clip `0.5`.  Raw pair and identity losses are mean patch Euclidean distance
divided by `sqrt(C)`.  Norm preservation is weighted `0.1`.  NLL retains the
Stage-A tail/logdet regularizer ratio, but both the tail NLL and squared
logdet are normalized by their feature-count scale (`C` and `C^2`) before
they are combined with the distance losses.  This prevents a purely
dimensional loss-scale advantage from deciding the NLL-versus-pair test.
No broad weight sweep is authorized.

## Metrics and gates

Every object/arm records:

- held raw paired distance and raw 1-NN shifted-normal inflation;
- reduction versus `Baseline1` (legacy arm ID `V0_default`);
- clean support LOO delta in V0 LOO standard deviations;
- latent standard-deviation ratio, effective-rank ratio, and neighbor identity
  retention;
- synthetic local-defect gap retention;
- invertibility, finite gradients, and deterministic repeat;
- per-epoch decomposed loss and a deterministic 32-pair held raw-distance
  probe; expensive full held 1-NN, structure, and defect metrics are computed
  once after each arm;
- effective rank and neighbor retention use the same deterministic sample of
  at most 2,048 support-latent rows.

Shifted-normal inflation is the increase over the clean support self-match on
the same support-LOO standardized score scale:

```text
((mean NN distance - LOO mean) / LOO std) - ((0 - LOO mean) / LOO std)
= mean NN distance / LOO std
```

It is therefore non-negative and the 10% relative-reduction gate has a valid
positive reference denominator.  Identity members of the held transform
mixture contribute zero inflation by construction.

Stage -1 passes only when one of `V2` or `V3` satisfies all of:

- macro shifted-normal inflation reduction at least 10%;
- improvement on at least 3/4 objects;
- absolute clean LOO delta at most 0.05 V0 standard deviations on every object;
- latent standard-deviation ratio in `[0.8,1.25]` on every object;
- effective-rank ratio at least 0.90 and neighbor retention at least 0.90;
- defect retention macro at least 0.97 and every object at least 0.90;
- reconstruction error at most `1e-5`, float32 logdet cancellation error at
  most `2e-4`, and deterministic/finite validity pass;
- `C2_random_pair` does not pass the same gate.

The frozen `V0` validity row is the experiment-wide sentinel. Each candidate
and control must still pass its own validity and safety gate, but a failing
candidate/control invalidates only that arm; it does not erase otherwise
interpretable measurements from valid arms.

Interpretation controls:

- if only `V1` passes, attribute the gain to augmentation exposure rather than
  paired canonicalization;
- if shift improves but defect retention fails, classify as anomaly
  absorption and hard-stop this loss branch;
- if neither raw-pair arm passes, stop the loss-canonicalization branch;
- if one raw-pair arm passes, the only authorized continuation is an all-eight
  support-only replication before any TESTpublic access.

## Execution and visibility

Run only on dsba5 in the fixed project container after verifying the selected
host GPU is idle and mapped to the container CUDA slot.  The runner prints and
writes a compact row after every epoch and completed object:

```text
object | variant | shift_reduction | loo_delta | std_ratio |
effective_rank_ratio | neighbor_retention | defect_retention | gate
```

Artifacts are written as `progress.jsonl`, per-object summaries, aggregate
verdict, access ledger, immutable source hashes, and compact checksums.  Dense
anomaly maps are not produced in Stage -1.
