# Shift-Conditioned Patch-Local Flow Report

Date: 2026-08-04  
Verdict: `EARLY_VALIDITY_KILL / KILL_FOR_CLAIM / NO_PUBLIC_ACCESS`

## Outcome

The task-specific register learned the photometric-shift prediction proxy, but
its Flow conditioning moved held shifted-normal patches in the wrong direction.
Correct-register conditioning increased raw-Euclidean 1-NN score inflation by
`20.41%` on average relative to E0, and it lost to both zero and shuffled
register controls on all four objects. The proposed separation
`shift information in g, shift removed from z` therefore was not established.

The frozen support-only controller stopped before enumerating or opening
`test_public`, `test_private`, or ground truth. No public metric, paper claim,
or all-eight continuation is authorized.

## Negative Evidence Intake and Design Corrections

The experiment was not treated as another all-token Transformer Flow. It
retained the patch-local MLP Flow and used a global register only through a
bounded residual FiLM path. Before implementation, the proposal was corrected
in four places:

1. The original zero-initialized residual head plus zero-initialized gain was a
   dead double gate. The implementation uses a zero-output residual head and a
   single nonzero bounded gain (`0.05`, cap `0.1`), preserving exact initial E0
   geometry while allowing gradients.
2. The scorer remains the repository's raw-Euclidean latent 1-NN contract;
   latent L2 normalization is used only inside consistency/neighborhood losses.
3. Shift prediction uses the identifiable relative descriptor
   `g(T(x)) - g(x)` with six multi-label transformation targets and masked
   continuous regression, rather than predicting a shift from an absolute
   descriptor alone.
4. Stage B explicitly retains dimension-normalized Flow NLL in addition to
   invariance, shift, neighborhood, identity, and residual losses.

The immutable contract is in `preregistration.md`.

## Implemented System

- `src/flow_tte/shift_conditioned.py`
  - learned four-query cross-attention register extractor
  - 64-dimensional shift descriptor
  - frozen Stage-A patch-local coupling Flow
  - per-layer FiLM residual scale/shift heads and bounded gain
  - conditional forward/reverse path and component losses
- `scripts/run_shift_conditioned_patch_flow.py`
  - exact fixed-16 support custody
  - DINOv3-H+/16 layers `[7,15,23,31]`, `layer_norm_mean`
  - one fitted DVT `position_mean`, alpha `1.0`, reused for all arms
  - Stage A and paired photometric Stage B
  - E0/E1/E5, correct/zero/shuffled register controls
  - support-only validity, held shift, identity, synthetic-defect, and shift
    prediction diagnostics
  - public phase protected by a machine-readable mechanism-pass sentinel
- `scripts/aggregate_shift_conditioned_patch_flow.py`
  - fail-closed mechanism and public gates
- `scripts/run_shift_conditioned_patch_flow_remote.sh`
  - fixed dsba5 container/GPU binding, immutable snapshot, preflight, background
    controller, checksum pullback, and guarded cleanup
- focused tests in `tests/test_shift_conditioned_flow.py`,
  `tests/test_shift_conditioned_patch_flow_runner.py`,
  `tests/test_aggregate_shift_conditioned_patch_flow.py`, and
  `tests/test_run_shift_conditioned_patch_flow_remote.py`

The runner installs the repository's PyTorch PyTree compatibility shim at
process startup. This was required because `torch.optim` indirectly imports
`transformers` through `torch._dynamo/torch.onnx` in the fixed container, not
only through the DINO wrapper.

## Local and Fixed-Container Verification

Local verification after the final implementation:

- focused pytest: `27 passed`
- Ruff: passed
- shell syntax: passed
- runner/aggregator Python compilation: passed
- core BasedPyright: `0 errors, 0 warnings, 0 notes`
- dependency-free runtime smoke: conditional nontrivial inverse maximum error
  `1.1921e-7`, logdet cancellation error `0`, checkpoint score roundtrip error
  `0`

The fixed container preflight verified:

- container: `hun_fsad_tta_012`
- dsba5 host GPU `3` mapped to container `cuda:0`
- GPU UUID: `GPU-24ac0286-7e02-dc5f-e320-48fddaa2c207`
- exactly one visible CUDA device
- support manifest SHA-256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`
- offline DINOv3 cache, imports, data paths, disk, compiled runner, runtime smoke,
  and source hashes

Pre-scientific attempts v1--v4 stopped before feature extraction: missing
runtime pytest fallback handling, missing `transformers`, and the PyTorch
PyTree compatibility boundary were repaired in sequence. Their source
snapshots were removed after v5 completion; small preflight failure records
remain marked invalidated where a result root existed.

## Remote Run and Custody

Run: `shift_conditioned_patch_flow_20260804_v5`  
Remote result root: `/workspace/results_remote/shift_conditioned_patch_flow_20260804_v5`  
Pulled result root:
`results/remote_runs/dsba5/shift_conditioned_patch_flow_20260804_v5`

The run completed in approximately 36 minutes. Every file listed in
`compact_artifact_sha256.txt`, including all four summaries and checkpoints,
passed local SHA-256 verification after pullback. The v5 immutable source
snapshot was then removed. No dense public anomaly maps were created.

The access ledger records exactly 16 allowlisted `train/good` support images
per object and asserts:

- `test_public_enumerated=false`
- `test_public_opened=false`
- `test_private_enumerated=false`
- `test_private_opened=false`
- `ground_truth_used_by_training_or_adaptation=false`
- `query_adaptation=false`

## Support-Only Mechanism Results

### Held shifted-normal score inflation

Lower is better. Reduction is `(E0 - correct) / E0`.

| Object | E0 inflation | Correct register | Zero register | Shuffled register | Reduction |
|---|---:|---:|---:|---:|---:|
| fruit_jelly | 0.7086 | 0.9609 | 0.5237 | 0.5185 | -35.60% |
| walnuts | 0.5744 | 0.6873 | 0.2914 | 0.2944 | -19.64% |
| vial | 0.8320 | 1.0426 | 0.8256 | 0.7927 | -25.31% |
| fabric | 0.4000 | 0.4045 | 0.2291 | 0.1967 | -1.10% |
| **Macro** | — | — | — | — | **-20.41%** |

Frozen requirements were at least `+5%` macro reduction and correct-register
wins over both nulls on at least 3/4 objects. Observed values were `-20.41%`
and `0/4`.

### Identity and local-defect retention

| Object | Abs. LOO mean delta / E0 std | Latent std ratio | Defect gap retention |
|---|---:|---:|---:|
| fruit_jelly | 0.0463 | 0.9724 | 0.5396 |
| walnuts | 0.0772 | 0.9823 | 0.6846 |
| vial | 0.0736 | 0.9247 | 1.6078 |
| fabric | 0.1380 | 1.0132 | 1.0053 |

Identity required every LOO delta `<=0.05`; only `fruit_jelly` passed. Latent
standard-deviation ratios all stayed inside `[0.8,1.25]`, so this is a local
geometry/memory displacement rather than a simple total-variance collapse.

Defect retention required macro `>=0.95` and every object `>=0.90`. The macro
was `0.9593`, but `fruit_jelly` and `walnuts` retained only `0.5396` and
`0.6846`; the per-object safety condition therefore failed. `vial` amplified
the synthetic defect while the two prior Transformer failure classes lost it,
showing strong object heterogeneity rather than a stable canonicalization.

### Shift prediction and training dynamics

The shift probe succeeded on every object:

| Object | Held predictor loss | Zero-predictor loss |
|---|---:|---:|
| fruit_jelly | 0.4179 | 0.8263 |
| walnuts | 0.4687 | 0.8314 |
| vial | 0.3517 | 0.8305 |
| fabric | 0.4043 | 0.8337 |

Across objects, the mean Stage-B shift loss fell from `0.7172` in epoch 1 to
`0.3393` in epoch 10, and neighborhood KL fell from `0.8329` to `0.5188`.
However, identity MSE rose from `0.00437` to `0.05368`, while the unweighted
residual magnitude reached `6.6742`. Thus the register encoded the augmentation
but the weakly regularized residual Flow used that signal in a way that moved
the latent memory and harmed anomaly contrast.

### Conditional inverse validity nuance

Exact E0/E1 parity, finite gradients, deterministic repeat, descriptor witness,
raw-Euclidean scoring, and reconstruction all passed. Maximum reconstruction
error was `3.8147e-6` under the frozen `1e-5` limit. Post-training float32
forward/reverse logdet cancellation was `3.0518e-5` to `1.2207e-4`, exceeding
the frozen `1e-5` limit, so the formal verdict is `EARLY_VALIDITY_KILL`.

This numerical failure is not the sole basis for stopping. Even if the logdet
tolerance were relaxed after the fact, the method would still fail the
independent shift (`0/4`, negative macro reduction), identity, and per-object
defect-retention gates. The scientific conclusion is therefore robustly
`KILL_FOR_CLAIM`, not merely an implementation-tolerance artifact.

## SuperAD Baseline and Unified Metrics

No TESTpublic run was authorized, so this experiment has no pAUROC, F1,
AU-PRO, gap-z, or SuperAD-comparable performance result. Historical baseline
numbers must not be attached to this new checkpoint as if they were matched
results. The reduced H+ support-only experiment is mechanism evidence only.

## Continuation Assessment

The preregistered one-time diagnostic continuation required correct-register
wins on at least two objects with retention preserved. It achieved zero wins
and failed per-object retention, so E2--E4 ablations, four-object public smoke,
all-eight evaluation, and random-support seeds are all blocked.

The evidence supports a narrower future hypothesis: the register can encode
photometric augmentation, but should not currently condition patch geometry.
If revisited under a new preregistration, the most defensible uses are score
calibration, conditional DVT strength, or support routing, with a substantially
stronger identity/residual constraint. That would be a new experiment rather
than a continuation of this failed branch.

## Conclusion

The proposed architecture preserved patch locality and successfully learned a
shift descriptor, but failed the causal requirement that the correct descriptor
reduce shifted-normal false positives without attenuating local anomaly
evidence. The correct register was consistently worse than its null controls,
and the two historically fragile classes again lost synthetic defect contrast.
The branch is closed for a performance claim and no public data was accessed.

## Post-Conclusion Storage Cleanup

- v5 results and four checkpoints were checksum-verified and retained locally.
- the v5 remote source snapshot was removed after verified pullback
- v1--v4 preflight-only source snapshots were removed with guarded checks that
  no scientific phase had started
- no dense public maps existed, so no map cleanup was required
- the compact remote result record remains available for audit
