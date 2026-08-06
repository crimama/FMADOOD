# DINOv2-L Basic default core-mechanism Stage M1

Status: `COMPLETE / KILL_FOR_CLAIM / NO_PUBLIC_ACCESS`

## Question and claim boundary

This experiment tests why the fixed DINOv2 ViT-L/14 Basic configuration
(`layers=[5,11,17,23]`) works under support-only synthetic distribution shift.
It decomposes DVT, Flow, density, and multi-layer fusion with exact 16-shot
references and no TTA, query update, memory expansion, TESTpublic, or
TESTprivate access.

The implemented core assay contains A0--A7 and L5/L11/L17/L23. Same-condition
SuperAD-16/RN-FMLK and the P1/P2 guided/morphology output chain are not part of
this run. Therefore it cannot support superiority or output-stage claims;
postprocessing remains `NOT_EVALUATED_MISSING_P1_P2`, and Stage M2 remains
unauthorized.

## Frozen population and execution

- Exact support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Four source-disjoint rotations per object: 12 fit / 4 held out.
- Conditions: C/S/D/SD; four shift axes, both signs, magnitudes x1/x2; four
  deterministic feature-visible defect families.
- dsba3 fixed container `hun_fsad_tta_012`, GPUs 0--3, two objects per GPU.
- Primary metrics: DINO feature patch-grid pAUROC@0.05 and fit-clean q=.995
  fixed-threshold F1. These are not original-image pixel metrics and are not
  directly comparable with historical pixel-level benchmark tables.
- Uncertainty: 2,000 paired hierarchical bootstrap replicates.

## Pre-metric attempts

- `v1`: stopped during CPU smoke because PyTorch 2.1 exposed only the private
  PyTree registration API while the installed transformers package required
  the public API. Reused the repository's audited compatibility shim. No GPU
  job or scientific row was produced.
- `v2`: CPU smoke passed, but all four GPU groups stopped at real M0. Top-2
  neighbor indices matched exactly; chunked-vs-full CUDA `cdist` absolute
  score differences were `1.43e-6`, `1.14e-5`, `3.81e-6`, and `1.91e-6`.
  The CPU fixture retained `1e-6`; CUDA parity was registered at `2e-5` with
  zero index mismatches allowed. No M1 condition row was produced.
- `v3`: produced three valid partial object files but stopped before `walnuts`
  rows because its M0 CUDA score error was `3.0517578125e-05` with zero
  NN-index mismatches. Partial objects are not reused. The clean v4 rerun uses
  CUDA score tolerance `1e-4`, retains CPU `1e-6` and exact NN indices, and
  changes no scientific setting.
- `v4`: completed clean all-eight rerun
  `dinov2l_default_mechanism_stage_m1_core_20260719_v4`. All four GPU groups
  and eight immutable object shards completed. The original aggregator then
  rejected every valid chunk because it expected total `support_reads=32`;
  the runner correctly recorded 128 manifest content-hash reads plus 32
  decoded shard-image reads, total 160. The audit was corrected to validate
  all three counters separately, covered by a regression test, and rerun over
  the unchanged 130,560 scientific rows. The original failure and both
  aggregator hashes are retained under `recovery/`.

## Results

### Validity and custody

- 130,560 compact rows, 8,192 fully paired factorial cases, 8/8 objects,
  4/4 source-disjoint folds, and 12 arms completed.
- All object M0 checks had zero nearest-neighbor index mismatches. Maximum
  chunked/full CUDA score difference was `3.05176e-5`, below the registered
  CUDA-only `1e-4` tolerance; the CPU fixture stayed at `1e-6`.
- Zero TTA calls, query updates, TESTpublic reads, and TESTprivate reads.
  Memory and thresholds remained static.
- The strict decoded-RGB custody audit verified 128 clean sources with zero
  hash mismatches, 10,880 arm-consistent stimulus cells, and identical defect
  identity between every D/SD pair.
- All compact pull checksums passed. An independent local reaggregation with
  the local dataset root produced byte-identical `summary.json`,
  `causal_contrasts.tsv`, and `per_object_metrics.tsv`.

### Primary mechanism contrasts

Contrast signs are first arm minus second arm. Intervals are frozen 2,000-draw
hierarchical paired 95% bootstrap intervals.

| Contrast | SD pAUROC@.05 | SD fixed F1 | shifted-normal q.995 tail | Interpretation |
|---|---:|---:|---:|---|
| DVT total, A0-A1 | `+0.00985` `[+0.00185,+0.01807]` | `-0.09908` `[-0.16941,-0.03926]` | `+0.22644` `[+0.04223,+0.50360]` | ranking rises while calibration and shift tails worsen |
| Flow with DVT, A4-A5 | `+0.04099` `[+0.02361,+0.06566]` | `-0.00693` `[-0.08041,+0.06324]` | `+0.16747` `[+0.03009,+0.37959]` | ranking gain, no reliable threshold gain, tail risk |
| Flow without DVT, A6-A7 | `+0.02625` `[+0.01879,+0.03357]` | `+0.03424` `[+0.01833,+0.05185]` | `+0.00745` `[-0.02619,+0.04203]` | cleanest necessary Flow effect; all eight objects improve |
| Density, A0-A4 | `-0.00082` `[-0.00191,+0.00021]` | `-0.00434` `[-0.01012,+0.00010]` | `+0.01885` `[-0.00115,+0.04476]` | no material contribution |
| Fused identity vs single-layer envelope | `-0.01715` `[-0.03456,-0.00026]` | `-0.03483` `[-0.06601,-0.00538]` | `-0.31334` `[-0.83321,-0.01198]` | fusion loses defect metrics on every object despite lower tails |

The exact A0 default scored macro `0.82713/0.82049` D/SD patch pAUROC and
`0.46766/0.45037` D/SD fixed F1. Its shift retention was
`-0.00664/-0.01728` pAUROC/F1, shifted-normal absolute tail inflation was
`0.27579`, and nearest-support identity retention was `0.75358`.

### What the failed interventions reveal

1. **DVT is a ranking aid, not a shift corrector in this assay.** A0-A1 raises
   D and SD pAUROC on 8/8 and 7/8 objects, but lowers D fixed F1 on 7/8 and SD
   fixed F1 on 8/8. It also worsens pAUROC/F1 retention, q.995 normal tails,
   and nearest-neighbor identity (`-0.04884`, all eight objects). The recorded
   DVT transform increases mean clean-to-shift cosine displacement from
   `0.00435` to `0.00898`; the largest increases occur for exposure and white
   balance. Thus the pAUROC gain comes from score ordering/separation while
   absolute score calibration drifts in the wrong direction.

2. **Both sides must share the DVT coordinate system, but this consistency is
   not correction.** Support-only DVT A2 loses `0.16072` SD pAUROC and
   `0.30131` SD F1 versus A1; query-only DVT A3 loses `0.01616` and `0.08426`.
   Both one-sided arms fail, so their deployment is invalid. Symmetric A0
   avoids the severe mismatch yet still inflates tails; this is why the frozen
   label is `RANKING_ONLY_OR_MIXED`, not `COMMON_COORDINATE_ALIGNMENT` or
   `SHIFT_CORRECTOR`.

3. **Flow has a real representation effect, but DVT makes that effect
   calibration-sensitive.** Without DVT, Flow improves both SD pAUROC and F1
   on all eight objects, slightly improves neighbor identity (`+0.00767`), and
   shows no resolved tail increase. With DVT, Flow still improves ranking on
   all eight objects but no longer improves F1, worsens F1 retention on all
   eight, and inflates the normal tail. Geometrically it raises effective rank
   by `+9.40` while reducing pair-distance Spearman correlation by `-0.151`
   and latent RMS by `-0.164`: useful remapping is accompanied by scale and
   neighborhood distortion. The primary SD interaction interval crosses zero,
   so formal independent component credit is not established; significant
   negative retention and NN-interaction diagnostics also mean
   non-significance cannot be read as independence.

4. **Fixed fusion is destructive relative to the layer envelope.** The
   object-wise best single layer beats fused identity features in SD pAUROC on
   8/8 objects and in F1 on 7/8. Layer 17 is the macro dominant layer. Flow
   later recovers ranking from this weak fused geometry, so the default's
   success should not be attributed to multi-level synergy.

5. **Density is effectively inert at the frozen weight.** Its effects are
   below the material thresholds, with a small resolved D fixed-F1 loss and no
   shift correction. It is not the source of the default's ranking gain and
   cannot repair the DVT/Flow calibration failure.

### Where degradation concentrates

- Exposure is the main distribution-shift failure: A0 retention is
  `-0.0229/-0.0418` pAUROC/F1; DVT adds `+0.7111` tail inflation with no SD
  pAUROC gain and `-0.1120` F1. White balance is second, with DVT tail
  `+0.1198` and F1 `-0.1043`.
- Increasing shift magnitude from 1 to 2 doubles the DVT tail effect
  (`+0.1528 -> +0.3001`) and deepens its F1 loss
  (`-0.0954 -> -0.1028`).
- Chromatic blotch exposes the sharpest calibration split: DVT changes SD
  pAUROC/F1 by `+0.0041/-0.2128`; Flow with DVT changes them by
  `+0.0309/-0.0977`. In contrast, Flow without DVT remains non-harmful and is
  strongly helpful for blur and local warp.
- The largest object-level DVT F1 losses are sheet_metal `-0.2802` and rice
  `-0.1974`; vial is the only object with a DVT SD-pAUROC loss (`-0.0061`).

### Structural implication

The operational default is best described as a **rank-improving but
calibration-unstable stack**. DVT and Flow can improve pAUROC while moving
shifted normal scores, nearest-support identity, and fixed-threshold decisions
in the wrong direction. The most promising tested simplification is the raw
fused Flow arm A6: relative to A0 it gives up `0.00853` SD pAUROC but improves
SD fixed F1 by `0.10161`, reduces absolute tail inflation from `0.27579` to
`0.04454`, and raises neighbor retention from `0.75358` to `0.80242`.

A follow-up structural design should therefore start from A6, remove density,
and repair the already-observed destructive fixed fusion before adding any new
nuisance correction. DVT should not be retained as a default shift module
unless a support-only objective explicitly constrains shifted-normal tail,
fixed-threshold retention, and neighbor identity. Routing, query adaptation,
and TTA remain excluded.

## Verdict

`MECHANISM_AUDIT_COMPLETE`, with component labels
`DVT=RANKING_ONLY_OR_MIXED`, `FLOW=MIXED_OR_INCONCLUSIVE`,
`DENSITY=NO_MATERIAL_CONTRIBUTION`, and
`FUSION=DOMINANT_LAYER_OR_DESTRUCTIVE_FUSION`.

The scientific claim remains `KILL_FOR_CLAIM`: these are support-only
synthetic patch-grid results, not original-pixel results. Same-condition
SuperAD/RN-FMLK and P1/P2 output arms were not executed, so superiority,
postprocessing, public-shadow, and Stage-M2 claims remain unauthorized.
