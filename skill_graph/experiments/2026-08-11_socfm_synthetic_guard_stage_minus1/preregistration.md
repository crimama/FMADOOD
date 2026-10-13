# SO-CFM Synthetic Guard Stage -1 Preregistration

Date: 2026-08-11
Maximum verdict: `PROMISING_DIAGNOSTIC`
Pre-run verdict: `SUPPORT_ONLY_DIAGNOSTIC / NO_CLAIM`

## Frozen scope and custody

- target: MVTec AD2 single-image, exact frozen support-16 only
- support manifest:
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`
- support manifest SHA-256:
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`
- objects: `can`, `fabric`, `vial`, `walnuts`
- folds: SHA-ordered four-fold leave-one-support-image-out, 12 fit / 4 held
- backbone: frozen DINOv2 ViT-L/14, extraction contract `[5,11,17,23]`,
  resolution 672
- diagnostic readout: layer 11 only, frozen before execution from prior
  depth-specific drift/retention evidence; row L2 normalization
- forbidden access: `test_public`, `test_private`, `ground_truth`, filename
  condition tokens, query-conditioned fitting, memory updates
- `comparable=false`; all SuperAD and RN-FMLK metrics are `N/A`

## Frozen execution shards

- local GPU 0: `can,fabric`, `CUDA_VISIBLE_DEVICES=0`, internal `cuda:0`
- dsba3 host GPU 3: `vial`, `CUDA_VISIBLE_DEVICES=3`, internal `cuda:0`
- dsba5 host GPU 3: `walnuts`; the fixed container exposes it as internal
  `cuda:0`

No object may migrate after metrics are visible. Resource failure permits only
an exact-config rerun on the same object; no positive method verdict is allowed
without complete four-object coverage.

## Normal trajectories

Training nuisance axes are exposure, gamma, contrast, and reciprocal red/blue
white balance at the existing CF-NTR amplitudes and both signs. Actual DINO
features at `t={0,0.5,1}` supervise midpoint finite-difference velocity targets.
Held normal shifts use doubled-amplitude single axes. The axis/sign code is a
controlled oracle in this mechanism assay and is shared across the image; this
run does not test query-side nuisance-code estimation.

## Arms

- `A0_raw`: position-aligned raw support feature-distance baseline
- `A1_orbit_bank`: position-aligned discrete transformed-support orbit bank
- `A2_socfm_normal`: normal-trajectory flow-matching field
- `A3_socfm_guard`: A2 plus synthetic anomaly-persistence guard
- `A4_socfm_anomaly_positive`: falsification arm that incorrectly treats
  synthetic anomalies as positive flow targets
- `N1_mask_shuffle`: A3 with the training defect mask spatially rolled away
  from its true support-side defect
- `N2_velocity_shuffle`: velocity targets permuted across images and axes
- `N3_patchwise_code`: explicitly not executed. A free per-patch code is a
  non-identifiable anomaly-absorption upper bound, not a valid shared-nuisance
  support-only comparator.

## Synthetic anomaly separation

Guard-training families are CutPaste and chromatic blotch. Held families are
local thin scratch and broad Gaussian blur. Seeds, source regions, locations,
and shapes are SHA-derived. Masks are known only for this synthetic assay. Held
families never contribute to fitting, model selection, or loss-weight
selection.

## Reduced diagnostics in this shard

- held shifted-normal patch coverage, defined as the fraction of aligned
  patches whose candidate support score is lower than A0, and p95 tail reduction
- local and broad held synthetic-defect residual-energy retention
- causal margins versus the discrete bank and N1/N2 hard nulls
- A3 versus A2 protection benefit and A4 anomaly-absorption stress behavior
- finite/nonconstant coverage, support/fold/transform parity, hashes, runtime,
  peak memory, and access ledger

Not applicable in this reduced launch: clean pairwise-distance Spearman, clean
nearest-neighbor identity retention, image-level nuisance-code drift, and
`N3_patchwise_code`. These must be reported as skipped/not_applicable, never as
executed metrics.

## Frozen continuation gates

All unrounded macro values must pass:

1. held shifted-normal displacement coverage `>= 0.50`;
2. A3 shifted-normal p95 tail reduction versus A0 `>= 0.20` and strictly
   exceeds A1;
3. A3-minus-N2 displacement-coverage margin `>= 0.20`;
4. held local and broad defect retention are each `>= 0.95`;
5. A3 weaker-family retention is not below A2 and A3 loses no more than `0.02`
   absolute shifted-normal tail reduction versus A2;
6. A3-minus-N1 held-defect retention margin `>= 0.03`;
7. A3 weaker-family retention is not below A1;
8. A3 weaker-family retention exceeds A4 by at least `0.05`, exposing the
   anomaly-positive absorption stress;
9. all artifacts are finite, nonconstant, complete, hash-audited, and confirm
    zero public/private/ground-truth access.

## Verdict semantics

- all gates pass: `PROMISING_DIAGNOSTIC`; permits one separately preregistered
  full-population shadow, never `KEEP`
- mechanism specificity survives but a safety/strength gate fails:
  `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`, with exactly one targeted follow-up
- A1 or a hard null absorbs the gain, defect retention fails materially, A4
  dominates, or custody fails: `KILL_FOR_CLAIM / NO_CONTINUE`
- implementation or artifact contract failure: invalid run; only exact-config
  repair/rerun is permitted

No threshold, loss weight, layer, transform amplitude, anomaly family, or object
may be tuned after any metric is opened.
