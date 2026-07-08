# FlowTTE vNext: NF Scoring With Test-Time Adaptation

Date: 2026-07-07

## Name

Keep the method name as **FlowTTE**.

Use the current implementation as `FlowTTE-LatentBank` in internal notes when a
distinction is needed. The next method branch is `FlowTTE-NFScore-TTA`.

## Motivation

The current FlowTTE implementation trains a Normalizing Flow on few-shot normal
support patches, freezes it, maps patches into latent space, and then expands a
latent memory bank. This is useful as a first prototype, but it still relies on
kNN-style distance ranking after projection.

The revised direction is to make the NF itself the scoring module:

- anomaly score should come directly from NF density, mainly patch NLL;
- test-time adaptation should update the normal density model or lightweight
  adaptation parameters, not only append latent vectors to a memory bank;
- memory expansion, if retained, should act as support for stable adaptation
  rather than the primary scoring mechanism.

## Current FlowTTE-LatentBank

1. Extract DINOv2 patch features from few-shot normal support.
2. Train NF once on support patches.
3. Freeze NF during test-time.
4. Transform support and query patches into latent `z`.
5. Select query patches with `NLL <= density_threshold`.
6. Absorb selected query `z` into a reservoir memory bank.
7. Score patches mostly by query-to-memory latent distance.

Key limitation: the NF is used as a projection/gating module, not as the main
anomaly scoring model. The learned density does not adapt after support fitting.

## Revised FlowTTE-NFScore-TTA

1. Extract patch features and optional spatial/context features.
2. Train an initial NF density model on few-shot normal support.
3. Score patches directly with NF likelihood:
   - patch score: `NLL(z) = -log p(z) - log |det J|`
   - image score: top-percent aggregation over patch NLL
   - pixel map: resized patch NLL map
4. Select pseudo-normal test patches using conservative density and spatial
   consistency gates.
5. Adapt the NF at test-time using selected pseudo-normal patches.
6. Re-score future test samples with the adapted NF.

The core change is:

`latent memory distance` -> `adaptive NF density score`

## DeCoFlow-Inspired Components To Reuse

Do not port DeCoFlow's full continual-learning protocol directly. Instead, use
the pieces that match FSAD-TTA.

### 1. NF as Scoring Module

Use patch NLL as the primary anomaly score. Distance-to-memory can remain only
as an auxiliary regularizer or fallback.

### 2. Tail-Aware Loss

Keep tail-aware training because normal tail coverage is central to preventing
ranking collapse. The current code already has a tail-aware loss path; the next
branch should make this loss part of both initial support training and
test-time adaptation.

### 3. Adapter-Only Test-Time Updates

Avoid full NF fine-tuning during TTA. Full updates can absorb anomalies and
destabilize the density manifold. Prefer adapter-only updates:

- freeze the base coupling subnet;
- add low-rank or residual adapter parameters to scale/shift subnets;
- update only adapters with pseudo-normal patches;
- keep the base flow as a density anchor.

This mirrors the useful DeCoFlow principle: coupling-layer subnet adaptation can
preserve flow invertibility and Jacobian validity when the external coupling
structure is unchanged.

### 4. Lightweight Alignment

Add a small feature alignment module before the NF:

- LayerNorm or feature standardization;
- learnable affine scale/shift;
- bounded parameterization for stability.

During TTA, update this alignment module before or alongside adapters. This is
lower-risk than updating the whole backbone or whole NF.

### 5. Residual Coupling Correction

If adapter-only updates are too rigid, add a small auxiliary coupling block
initialized as identity. During TTA, update only this block. This gives the
density model a controlled correction path without rewriting the base flow.

## TTA Safety Gates

Test-time adaptation must be conservative. Candidate pseudo-normal patches
should satisfy multiple gates:

- low NLL under the current NF;
- low local anomaly score relative to image distribution;
- stable under simple augmentations or neighboring patch context;
- optionally close to initial support statistics;
- never update on high-score tail patches from the test image.

Recommended first rule:

`candidate = NLL <= q_support(0.80 or 0.90)` and top anomaly regions excluded.

## Loss During TTA

Use pseudo-normal patches only:

`L_tta = TAL_NLL(pseudo_normal) + lambda_logdet * logdet_reg + lambda_anchor * anchor_reg`

Where `anchor_reg` keeps adapted parameters close to the support-trained base.
For adapter-only updates, `anchor_reg` can be L2 on adapter parameters or KL/NLL
stability on cached support features.

## Expected Benefit

This should address the current failure mode more directly:

- current method controls memory volume but does not update density;
- vNext controls the density manifold itself;
- ranking collapse is handled by adapting the normal likelihood boundary, not
  only by compressing memory-bank distances.

## First Implementation Branch

Minimal branch before any full experiment:

1. Add a config switch: `score_mode = latent_distance | nf_nll`.
2. Implement static NF-NLL scoring without TTA.
3. Run AD1/AD2 static ablation against the current latent-distance score.
4. Add adapter-only TTA after the static NF score is verified.
5. Compare:
   - current `FlowTTE-LatentBank`;
   - `FlowTTE-NFScore` static;
   - `FlowTTE-NFScore-TTA` adapter-only.

Hard gate: do not run a full AD2 sweep until the static NF-NLL score improves
or at least matches the current AD1 reduced/full metrics.

## 2026-07-07 Static NLL Gate Result

The first static raw-NLL diagnostic was run on MVTec AD2 `can,rice`, 4-shot.

- `FlowTTE-NFScore` static NLL: mean AUROC/F1 `0.674336` / `0.127093`
- Previous `FlowTTE-LatentBank`: mean AUROC/F1 `0.765772` / `0.315589`
- Delta vs LatentBank: `-0.091436` AUROC / `-0.188495` F1
- Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_STATIC_NLL`

Do not scale raw static NLL to full AD2. Any continuation must change the
scoring/adaptation mechanism, for example support-relative NLL calibration or
adapter-only TTA with an explicit support anchor and a latent-distance control.
