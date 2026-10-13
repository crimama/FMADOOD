# TARCD-LoRA-S3-VR Stage -1 Preregistration

Date: 2026-08-26  
Protocol: `tarcd-lora-s3-vr-stage-minus1-support-adapted-20260826`  
Status: `PREREGISTERED / NO S3 RESULTS INSPECTED`  
Seed: exactly one, `20260826`

## Negative Evidence Intake

TARCD-LoRA-S2-DCR is frozen with verdict
`KILL_FOR_CLAIM / NO_CONTINUE`. Its four-object support-only panel was valid in
`12/12` held cells, but correction cosine, hard-null superiority, complete-cell
success, and object success were respectively `0/12`, `0/12`, `0/12`, and
`0/4`. The D2 held-cell median correction cosine was `0.3250056598`, median
norm ratio was `0.7885586791`, and the median of object hard-null margins was
`-0.1505263691`. A static fit-support mean correction was the strongest null in
`10/12` cells.

S2 also exposed a previously unisolated numerical failure. At zero-init LoRA,
the predicted correction is exactly zero, so its cosine term uses the `1e-8`
denominator branch. The D2 first pre-clip parameter-gradient norms were
`10,941,915` (`can`), `17,676,164` (`fabric`), `22,403,718` (`vial`), and
`12,608,531` (`walnuts`), after which every step was globally clipped at `1.0`.
The log-norm term has zero derivative at a zero predicted norm. Thus S2 did not
cleanly test direct-correction identifiability independently of zero-start loss
conditioning.

S2's repeated `can` sentinel and panel had identical scientific metrics but
non-identical adapter bytes for several trained arms. Before any S3 result is
allowed to license a panel or a successor, S3 therefore adds a validation-only
repeatability prerequisite: the complete `can` gate is rerun sequentially on
the same physical GPU and immutable snapshot. This does not alter training or
the candidate mechanism.

S3-VR is a separately named causal ablation, not an S2 retune. It changes only
the correction-vector loss. It does not change LoRA placement/rank, optimizer,
learning rate, weights, RN guard, synthetic generators, shift families, data,
schedule, seed, thresholds, evaluation, or control arms. S1 and S2 source and
artifacts remain frozen.

## Claim, Evidence, Boundary, and Positioning

- Claim: replacing the zero-start singular direction/log-scale objective with
  one bounded-gradient vector-regression objective can determine whether S2's
  poor held correction was caused by loss conditioning rather than by the
  shifted-assumed-plus-synthetic supervision itself.
- Evidence: finite nonzero zero-start gradients with an analytic bound; at
  least 100-fold first-gradient reduction versus paired S2 objects; unchanged
  custody/no-op controls; and held correction/RN/synthetic/null metrics under
  the exact S2 evaluation.
- Boundary: one-seed, four-object, support-adapted mechanism diagnostic only.
  This is not a segmentation benchmark and cannot support a SuperAD claim.
- Positioning: the first item in a sequential repair ladder. No factorized
  head, AGC, layer change, RN redesign, generator change, or synthetic
  quarantine is included.

## Single Frozen Change: Normalized Vector Huber

For predicted correction vector `v`, detached target `v*`, and the unchanged
object-specific activity scale `tau`, define

`s(v*) = stopgrad(max(||v*||_2, tau))`

and

`D_VR(v,v*) = sum_j smooth_l1((v_j-v*_j)/s(v*), 0; beta=1)`.

The same replacement is used for clean correction, anomaly correction, and
anomaly recomposition. Existing anomaly-mask balancing is unchanged. Identity
and slack-only RN losses and all scalar coefficients are unchanged.

At zero start, `||v*||/s <= 1`, so every coordinate is inside the quadratic
Huber region and

`D_VR(0,v*) = 0.5 ||v*||^2/s^2 <= 0.5`,

`||grad_v D_VR(0,v*)||_2 = ||v*||/s^2 <= 1/tau`.

This removes the cosine `1/eps` direction gradient and the zero-derivative
log-scale branch without introducing a warm-up, schedule, or second loss.

## Frozen Matched Design

- Target: MVTec AD2 single-image at
  `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `can`, `fabric`, `vial`, `walnuts`.
- Fit support: exact 16 `train/good` paths per object from
  `skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json`,
  SHA-256
  `7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3`.
- Evaluation: the same four deterministic support-disjoint `train/good` bases,
  three held shift cells, and held synthetic generator as S2.
- Backbone/adapter: frozen DINOv2 ViT-L/14, fused-qkv rank-4 alpha-8 LoRA in
  zero-based blocks 8--11, zero-init B, unchanged bf16-forward/fp32-loss path.
- Training: exactly `144` AdamW steps per trained arm, learning rate `1e-4`,
  zero weight decay, global clip `1.0`, no early stopping.
- Randomization: retain the S2 randomization namespace
  `tarcd-lora-s2-dcr-stage-minus1-support-adapted-20260826` so schedules,
  masks, shifts, generators, and shuffled targets are matched exactly. Output
  protocol/schema names are S3-only and cannot be confused with S2.
- Arms, coefficients, RN slack, hard nulls, generator/confound probes, and all
  cell thresholds are byte-for-byte/config-equal to S2 except for the explicit
  `correction_loss_contract` metadata.

## Data Custody

Only exact `train/good` fit supports, sealed support-disjoint `train/good` held
bases, and deterministic descendants may be opened. `TESTpublic`, private test,
ground truth, query images, query memory updates, and public authorization are
forbidden. Every result must disclose false forbidden-access flags and an empty
`forbidden_kinds_opened` ledger. No dense `anomaly_maps/` are required or
retained.

## Gates

### Validity and Conditioning

- finite nonconstant support diagnostics and held maps;
- exact zero-init no-op and duplicate-RN controls within S2 tolerances;
- finite positive `tau` and finite losses/gradients;
- synthetic zero-start vector gradient is nonzero and no larger than its
  analytic `1/tau` bound up to `1e-6` relative tolerance;
- before panel expansion, two sequential GPU0 `can` runs must have identical
  adapter-state SHA-256 panels and identical normalized object payloads; a
  mismatch stops the run before GPUs 0--3 are opened;
- the panel `can` result must also be bitwise identical to both prerequisite
  runs for any scientific continuation verdict;
- for every panel object, D2 first pre-clip parameter-gradient norm is positive
  and at most `0.01` times its frozen S2 value above.

Sentinel-to-panel expansion is authorized by custody/schema/finite/no-op
validity only. It does not use `can` scientific performance.

### Unchanged Strict Mechanism Gate

Every S2 held-cell condition remains unchanged: correction cosine/norm,
anomaly cosine/norm, clean and anomaly exact-RN topology, D2 hard-null margin,
synthetic contribution, RN-guard contribution, and q99 no-harm. A compact pass
is at most `PROMISING_DIAGNOSTIC / MULTISEED_REQUIRED`, never a claim.

### Single-Step Continuation Gate

A subsequent, separately named single change is licensed only if all four
objects are valid and conditioning passes, no more than one object has median
q99 reduction below `-5%`, and both hold:

- all-cell D2 median correction cosine is at least `0.4250056598`, exactly
  `+0.10` over S2;
- median object hard-null correction margin is at least `-0.1005263691`,
  exactly `+0.05` over S2.

If this gate passes but the strict mechanism gate fails, verdict is
`KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`. If conditioning fails or either
improvement floor fails, verdict is `KILL_FOR_CLAIM / NO_CONTINUE` for S3-VR;
there will be no loss-weight, beta, normalization, seed, rank, or learning-rate
sweep.

## Sequential Execution

1. Implement separately named S3 core, runner, aggregator, tests, and dsba3
   controller while preserving S2 files.
2. Pass `py_compile`, focused pytest, shell syntax/selftest, and deterministic
   synthetic support-only smoke with the analytic gradient proof.
3. Run one GPU0 `can` support-only validity sentinel in fixed container
   `hun_fsad_tta_012`, then repeat that exact gate sequentially on GPU0.
4. Only if both runs are valid and bitwise identical, run the four-object panel
   on host GPUs `0,1,2,3` from the same immutable snapshot.
5. Require panel `can` to repeat both prerequisite runs, pull compact artifacts,
   independently reaggregate, compare against the
   frozen S2 diagnostics, record the verdict, and verify no dense maps or
   forbidden access.

## SuperAD and Unified Metrics

`superad_seg_AUROC_0.05=N/A`, `method_seg_AUROC_0.05=N/A`,
`delta_vs_superad=N/A`, and `benchmark_comparable=false`. No Stage -1 result
automatically authorizes public evaluation.
