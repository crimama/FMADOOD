# Cross-Image-Calibrated Region Completion Stage -1 Report

Date: 2026-08-10

Verdict: `KILL_FOR_CLAIM / NO_CONTINUE`

Scope: four-object MVTec AD2 diagnostic; SuperAD/RN-FMLK `N/A`; `comparable=false`

## Negative Evidence Intake

This experiment replaced the invalid per-query rank fusion with a single score
calibration fitted only from image-disjoint P16 support folds. It did not reopen
the rejected feature-space two-band DVT, support-bank illumination augmentation,
normal-tail identification, or any per-image/query normalization. Current and
fixed low-pass DVT remained separately fitted score-map branches and were combined
only after scoring.

The tested question was deliberately narrow: after making the two branch scores
commensurable across query images, can current-DVT seeds use aligned low-pass
support to recover missing defect regions beyond calibrated max, generic
current-only reconstruction, and a marginal-preserving spatial roll?

## Implementable Design

For each object and each score branch, 16 true leave-one-support-image-out folds
fitted DVT, Flow, calibration scores, and latent memory on the other 15 images.
The held image never entered its fold's fit set or memory. The pooled held-image
patch scores fitted one strictly increasing empirical-tail map
`-log10 P_LOO(S >= s)` with fixed probability knots `0.001..0.999`.

The fixed candidate used calibrated current score `C` and low-pass score `L`:

```text
seed    = C >= 3
support = (L >= 2) OR seed
base    = max(C, L)
```

For each 8-connected support component touching a seed, the operator lifted the
component to `min(max_R C, max_R L)`. Thresholds, connectivity, lift, objects,
Flow, and scorer were frozen before endpoint evaluation.

The seven arms were raw current (`A0`), calibrated current (`A1`), calibrated
low-pass (`A2`), calibrated max (`A3`), current-only reconstruction (`A4`), the
candidate (`A5`), and half-grid rolled low-pass (`N1`).

## Evaluation Alignment

The experiment used MVTec AD2 single-image `test_public/good,bad` for
`can,fabric,vial,rice`, the exact SuperAD-selected P16 support manifest, frozen
DINOv3-H+/16 layers `[7,15,23,31]`, layer-normalized mean fusion, three-epoch
patch-local Flow, raw latent 1-NN, and density weight zero. No query update,
private split, condition token, file metadata, or ground truth was used for fit or
operator selection.

This is not a same-condition SuperAD comparison. The unified fields are fixed to
`superad_seg_AUROC_0.05=N/A`, `superad_seg_F1=N/A`, `delta_vs_superad=N/A`, and
`comparable=false`.

## Code Modification / Creation

- `scripts/run_flow_tte_phase3_scorer_suite.py`: opt-in exact P16 LOO score
  calibration emission with fold custody and Flow checksum evidence.
- `scripts/cross_image_region_completion.py`: frozen calibrator, seven arms,
  component operator, roll null, compact evaluator output, and map audits.
- `scripts/aggregate_cross_image_region_completion.py`: independent shard
  validity, calibration preservation, parity, null, guardrail, and verdict gates.
- `scripts/run_cross_image_region_completion_stage_minus1.sh` and
  `scripts/run_cross_image_region_completion_stage_minus1_remote.sh`: local and
  immutable-snapshot dsba5 execution with dense-map cleanup.
- `tests/test_flow_tte_phase3_scorer_suite.py`,
  `tests/test_cross_image_region_completion.py`, and
  `tests/test_aggregate_cross_image_region_completion.py`: LOO exclusion,
  monotonic calibration, fixed reconstruction, roll marginal, aggregation, and
  verdict regressions.

## Added Code Evaluation

- Related focused regression suite: `73 passed`.
- Final direct calibration/operator/aggregator suite: `35 passed`.
- Scoped Ruff, Python compilation, shell syntax, and `git diff --check`: pass.
- Independent preregistration-to-summary verifier: `PASS`.
- Runtime operator, Phase-3 scorer, and remote orchestration hashes matched the
  immutable dsba5 preflight snapshot.

## GPU and Remote Execution

- Local fixed container `hun_fsad_tta_local_stage1a0`, RTX 4090 GPU 0:
  `can,fabric,vial`.
- dsba5 fixed container `hun_fsad_tta_012`, host GPU 3 / container CUDA slot 0:
  duplicated `can` plus `rice`.
- dsba3 GPUs 0--3 were already occupied at submission and were not pre-empted.
- Remote preflight hashes:
  runner `0c63df9c5833daa1a62b5ce4204d830e667e68fc0bd7b398c650a755d9d2217f`,
  operator `5f4dfb7b5663e16fe73e8bc9c387898d456523ebe1255a50cf53cbeddef8803a`,
  Phase-3 scorer `3414042b9167c03e639be0a3f0e6a89e48f343a4061c03b14e89aec20d775652`.
- Support-manifest SHA-256:
  `9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5`.
- Duplicated `can` parity covered every shared continuous metric across all seven
  arms; maximum absolute difference was `5.46e-7`, below the frozen `1e-3`
  tolerance.

## Validity

All seven validity gates passed. Every requested arm and object had complete,
finite, nonconstant prediction/GT coverage. Both branches for all four objects
had exactly 16 folds, 15 fit images per fold, zero held/fit overlap, 999 strictly
increasing knots, and no public-query/private/metadata fitting. Held patch-score
counts were 61,152 (`can`), 33,600 (`fabric`), 38,304 (`vial`), and 33,600
(`rice`) per branch. Calibration-only metrics were preserved within `1e-3`, N1
preserved sorted low-pass marginals with nonzero rolls, and every Flow checksum
was unchanged by memory population.

## Results and Analysis

Macro absolute metrics:

| Arm | pAUROC | F1 | AU-PRO | Component recall | Pixel AP | Normal FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 current | .809188 | .456053 | .530087 | .532813 | .455591 | .003708 |
| A3 calibrated max | .805512 | .450537 | .525121 | .530208 | .446607 | .003437 |
| A4 current reconstruction | .808884 | .469779 | .525423 | .449479 | .388221 | .003439 |
| A5 CI-RC | .804885 | .458885 | .520431 | .449479 | .380085 | .003389 |
| N1 rolled low-pass | .758623 | .310373 | .432677 | .475305 | .236405 | .004156 |

Preregistered contrasts:

| Contrast | AU-PRO | Component recall | pAUROC | F1 | Medium AP | Small top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A5 - A0 | -.009657 | -.083333 | -.004303 | +.002833 | -.196325 | +.018061 |
| A5 - A3 | -.004691 | -.080729 | -.000626 | +.008348 | -.197124 | +.030339 |
| A5 - A4 | -.004993 | +.000000 | -.003999 | -.010894 | -.007082 | -.006748 |
| A5 - N1 | +.087754 | -.025826 | +.046262 | +.148513 | +.188919 | +.114721 |

The candidate did not create the required component-recovery signal. It lost
`-.083333` component recall and `-.009657` AU-PRO versus current DVT, while the
simple calibrated max also beat it on both co-primary endpoints. Current-only
reconstruction produced exactly the same macro component recall as A5 and
higher AU-PRO, so the aligned low-pass branch added no mechanism beyond generic
morphology. The rolled null was much worse on AU-PRO, showing that alignment
changes the map, but A5 still had lower component recall; this does not rescue
the preregistered two-endpoint claim.

The apparent `+.002833` F1 change versus A0 hides a categorical loss on `rice`
(`-.053521`), beyond the `-.030` no-harm limit. `can` normal FPR was `.008421`
against an allowed `.007780`. Medium AP fell `-.196325`; medium top-5 coverage
was nearly unchanged (`-.001074`), indicating score redistribution rather than
new region recovery. Small top-1 improved, but it cannot offset the co-primary,
normal-FPR, per-object, and medium-AP failures. Medium scale macros use only
eligible per-image scale rows before object-level macro averaging.

## Continuation Assessment

- Strict diagnostic gate: fail.
- Single-step continuation gate: fail.
- Calibrated-max recoverability ceiling: absent; A3 beats A5 on both co-primary
  endpoints.
- Current-only morphology null: absorbs component recall and exceeds A5 AU-PRO.
- Normal-FPR guard: fail on `can`.
- Object no-harm: fail on `rice` F1.
- Small/medium guard: fail through medium AP collapse.
- Frozen all-eight replication: not licensed.

## Conclusion

`KILL_FOR_CLAIM / NO_CONTINUE`. Cross-image P16 LOO calibration itself is valid
and reproducible, but the proposed seeded low-pass completion does not recover
additional defect components. It mainly redistributes/lifts already detected
regions, loses pixel AP and medium AP, and is absorbed by calibrated-max and
current-only controls. Do not tune the tail thresholds, connectivity, lift,
calibration knots, fusion, or object-specific policies; that would be post-hoc
retuning of a branch whose direct recoverability premise failed.

Primary artifact:
`results/cross_image_region_completion_stage_minus1_20260810/summary.json`
(`sha256=810675b4b28ca4eeced5c74ace7d72c2c5a0505c88349e38c977977215b45c9b`).

## Post-Conclusion Storage Cleanup

Both completed result roots contain no `anomaly_maps/` directory. Local dense
artifacts were reduced from about 5.7GB to 15MB and the pulled remote artifact is
9.8MB; compact JSON, manifests, calibration evidence, hashes, and logs remain.
The immutable dsba5 source snapshot was removed after report verification; the
remote compact result root remains available.
