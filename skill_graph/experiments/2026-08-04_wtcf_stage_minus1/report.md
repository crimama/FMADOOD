# WTCF Stage -1 Support-Only Report

Date: 2026-08-04  
Verdict: `EARLY_VALIDITY_KILL / NO_FULL_STAGE / NO_PUBLIC_ACCESS`

## Outcome

Witness-Targeted Cross-fit Flow (WTCF) was implemented and evaluated under the
frozen support-only protocol. The valid run stopped after the preregistered
`can/seed0` sentinel because the WTCF arm failed its directional JVP validity
gate. The other 14 object/seed cases, MVTec AD2 public/private test data, and
ground-truth data were not accessed.

WTCF found five nonzero witnesses, each supported by three distinct source
pairs and three fold pairs. It remained finite, preserved the frozen
function-space geometry, and maintained a faithful soft-to-hard 1-NN proxy.
It did not, however, produce the preregistered selective contraction: final
witness JVP ratios were `0.9236--0.9493` relative to A6, above the frozen
target ceiling `0.85`. Unsupported ratios stayed within their required
`0.90--1.10` range. The WTCF-specific mechanism is therefore not validated.

## Sentinel Validity Gate

| check | result | evidence |
| --- | --- | --- |
| support/test access custody | pass | 16/16 selected supports opened; zero test/label attempts |
| finite, nonconstant, complete arms | pass | all nine arms |
| soft/hard 1-NN proxy | pass | WTCF Spearman `.999996`, top-tail overlap `1.0`, q995 difference `.000015` |
| function-space geometry | pass | WTCF pair Spearman `.998262`, neighbor retention `.929688`, inverse error `9.54e-7` |
| nonzero stable witness | pass | 5 witnesses; source/fold-pair diversity 3 for every witness |
| directional JVP | **fail** | witness ratios `.9236--.9493`, target `<=.85` |
| training finite | pass | finite objective and state |

The controller consequently wrote `EARLY_VALIDITY_KILL_NO_FULL_STAGE` and did
not schedule `fruit_jelly`, `rice`, `fabric`, `sheet_metal`, or the remaining
seeds.

## Optimization Diagnosis

WTCF accepted only the first five parameter updates. The following 59
five-update blocks were rejected and both model and optimizer state were
restored. This is not a numerical crash: it is the frozen function-space
trust region rejecting proposals that would damage the A6 geometry.

The final WTCF state illustrates the constraint conflict:

- median witness JVP ratio: `.9369`;
- median unsupported JVP ratio: `.9828`;
- latent RMS ratio: `.9793`;
- effective-rank ratio: `.9963`;
- pair-distance Spearman: `.9983`;
- neighbor retention: `.9297`;
- score-standard-deviation ratio: `.9913`;
- coupling-logdet q99 ratio: `.8346`;
- maximum inverse error: `9.54e-7`.

Thus the implementation can preserve the global deployed 1-NN geometry, but
the current loss/update geometry cannot move far enough inside that feasible
region to reach the required witness contraction. `witness_only` showed the
same pattern (5 accepted updates, 59 rejected blocks, median witness ratio
`.9355`). `global_isometry` accepted 10 updates but retained a median witness
ratio of `.9460`. The repeated-residual witness identifies a direction, but
the frozen candidate does not demonstrate that the Flow can selectively
contract it while satisfying the trust region.

The unconstrained `crossfit_only` control moved in the opposite regime: it
accepted all 300 updates and strongly contracted every probed direction, but
failed geometry (pair Spearman `.3026`, neighbor retention `.5234`, latent RMS
ratio `.5245`, score-std ratio `.1859`) and failed the soft/hard q995 proxy.
Its detector metrics are therefore an invalid collapse-associated result, not
evidence that unconstrained cross-fit optimization solves the problem.

## Descriptive Metrics Only

The validity gate failed before performance adjudication, so these sentinel
synthetic C/S/D/SD metrics cannot support a detector claim or hyperparameter
selection. They are retained only to show why validity must precede apparent
performance.

| arm | held-normal q995 (A6 units) | pAUROC@.05 | fixed F1 | AU-PRO@.05 |
| --- | ---: | ---: | ---: | ---: |
| A6 | 1.8258 | .7753 | .4902 | .5624 |
| global isometry | 1.6396 | .7808 | .4896 | .5737 |
| random witness | 1.7472 | .7764 | .4920 | .5653 |
| witness only | 1.8692 | .7704 | .4758 | .5544 |
| WTCF | 1.7530 | .7760 | .4871 | .5649 |
| cross-fit only (invalid geometry) | -1.4512 | .8568 | .6470 | .7326 |

Relative to A6, WTCF changed the three metrics by only
`+0.00069/-0.00316/+0.00246` and did not descriptively dominate the global
isometry control. No keep/kill performance gate was evaluated because the
directional validity prerequisite failed.

## Implementation and Verification

The implementation adds image-disjoint cross-fit soft 1-NN optimization,
normal-residual witness discovery, JVP direction controls, full
model-and-optimizer rollback, exact hard 1-NN scoring, null arms, and a
validity-first remote controller. Focused local verification passed:

- 11 WTCF unit tests;
- dependency-free WTCF core smoke;
- Python compilation for the core, runner, and aggregator;
- remote DINOv2 cache/import/data/GPU preflight;
- shell syntax and whitespace checks.

Strict `basedpyright` remains open because the experiment module inherits
substantial dynamic NumPy/Torch typing debt; runtime tests and the valid remote
sentinel passed, but a clean strict-type claim is not made.

Runs v1--v5 were infrastructure-invalid and are excluded from the scientific
result: v1 lacked remote `pytest`; v2 attempted to load the intentionally
excluded credential preset from the snapshot; v3 reached evaluation but used
an unsized AU-PRO mask; v4 and v5 were invalidated before output after adding
optimizer-state rollback and the zero-witness/directional target gate. Their
compact failure logs were pulled locally. All v1--v6 isolated snapshots and
the invalid v1--v5 remote result directories were removed. The valid v6
compact result remains remote and was pulled locally; host GPU 3 returned to
1 MiB and 0% utilization.

## Decision

The frozen WTCF candidate is closed. It cannot be presented as a validated
method, cannot proceed to the remaining Stage -1 cases, and cannot access the
full MVTec AD2 benchmark. A successor must be a structural redesign rather
than a threshold or learning-rate retry. In particular, it must demonstrate a
nonempty intersection between selective witness contraction and the deployed
1-NN geometry trust region before any detector metric is interpreted.

Compact evidence:
`results/remote_runs/dsba5/wtcf_stage_minus1_20260804_v6/`.

