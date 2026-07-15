# Static Flow-LatentBank MVTec AD1 component 0/1/2 diagnostic

Date: 2026-07-13  
Verdict: `KEEP_DENSITY0_DIAGNOSTIC / KILL_CLS_SOFT_W10 / KILL_FIXED_GUIDED_R8_FOR_AD1_CLAIM / BLOCKED_BASELINE`

## 1. Negative Evidence Intake

The accepted MVTec AD1 4-shot static Flow-LatentBank setting is strongest with
DVT disabled. Every nonzero DVT alpha previously failed the all-five retention
gate, even when mean pixel localization improved. This experiment therefore
keeps DVT and TTE excluded and independently tests three non-DVT components
transferred from the AD2 method family.

## 2. Motivation

Determine whether density-score removal, frozen CLS-guided retrieval, or RGB
guided map refinement can improve the accepted AD1 setting without changing
its backbone, supports, flow, memory, or evaluator. Each arm changes exactly
one component; combinations and AD1 tuning are excluded.

## 3. Implementable Design

- Dataset: classic MVTec AD, all 15 classes and complete test split.
- Supports: first four `train/good` images, seed 0, identity only.
- Encoder: frozen/eval `dinov2_vitb14_reg`, shorter-edge 448, layers
  `[2,5,8,11]`, `layer_norm_mean` fusion.
- Flow: three epochs, two couplings, hidden multiplier 1, LR 2e-4, clamp 1.9,
  tail 0.3/top-k 0.05, lambda-logdet 1e-3.
- Memory: static latent 1-NN bank, `4096 -> 4096`, distance weight 1.0.
- Base scoring: density quantile 0.90, density weight 0.25, top-1% image score.
- Excluded: DVT, TTE, register conditioning, foreground prior, morphology,
  calibration, support augmentation, and resolution changes.

The independent arms are:

0. `density0`: density weight `0.25 -> 0.0` only.
1. `cls_soft_w10`: unconditional Flow remains unchanged; frozen CLS cosine
   soft penalty with fixed AD2 weight 10 is added only to memory retrieval.
2. `guided_r8`: unchanged base maps receive grayscale RGB guided filtering at
   half-native scale, radius 8, epsilon 0.01, with per-image de/min-max.

## 4. Evaluation Alignment

All arms use the same five class-macro metrics: i-AUROC, i-AUPRC, p-AUROC,
p-AUPRC, and p-AUPRO@FPR0.30. The accepted reference is
`0.972100416682/0.987431867365/0.973747446630/0.581160338468/0.938160366361`.

The pre-registered retention gate allows no metric to lose more than 0.10
percentage point. A positive diagnostic additionally requires at least +0.20
point in p-AUPRC or p-AUPRO. The regenerated identity maps for arm 2 had to
match the reference within `1e-9`.

The scheduler was changed after preregistration at the user's request: arm 2
moved from a waiting dsba3 four-GPU allocation to two class shards on dsba5
GPUs 0/1 after arms 0/1 completed. This changes only execution allocation, not
the locked method or evaluation.

## 5. Code Modification / Creation

- `fmad/backbones/dinov2.py`: frozen CLS context extraction adapter.
- `scripts/run_flow_tte_mvtec_ad1.py`: explicit context-source and
  memory-retrieval context arguments; default behavior remains no-context.
- `scripts/run_flow_tte_mvtecad1_guided_refinement.py`: guided-r8 post-hoc
  evaluator using the existing refinement implementation.
- `scripts/aggregate_mvtecad1_metric_chunks.py`: exact 15-class row merger and
  macro aggregation, avoiding averages of shard means.
- `scripts/run_mvtecad1_component01_remote.sh`: isolated arms 0/1 launcher.
- `scripts/run_mvtecad1_component2_guided_sharded_remote.sh`: resumable two- or
  four-GPU class sharding, identity parity, guided evaluation, and cleanup.
- Three focused test files cover CLS extraction, CLI contracts, guided map
  layout/finite output, and chunk aggregation.

## 6. Added Code Evaluation

- Focused tests before remote execution: 45 passed.
- Full regression suite after result pull: 424 passed, with five existing
  PyTorch transformer warnings.
- Python compilation, launcher shell syntax, and `git diff --check`: passed.
- Arm-2 identity aggregate matches the accepted reference exactly; maximum
  absolute difference across all five metrics is `0.0`.

## 7. Remote Execution and Audit

- Host/container: dsba5, existing `hun_fsad_tta_012` container.
- GPUs 0/1: `density0` and `cls_soft_w10` in parallel, followed by two
  disjoint guided-r8 class shards.
- Result roots:
  - `results/remote_runs/dsba5/flow_latentbank_mvtecad1_shot4_component01_20260713_v1`
  - `results/remote_runs/dsba5/flow_latentbank_mvtecad1_shot4_guided_r8_20260713_v1`

The audit passed all 45 class-runs. Every class used exactly supports
`000.png`--`003.png`; all memories stayed `4096 -> 4096`; DVT mode was `none`;
metrics were finite; all 15 classes appeared once in each aggregate; logs had
no traceback, OOM, or killed process; and local pulled artifacts contain zero
`.npy`/`.npz` dense maps. Cleanup evidence is retained for every run/shard.

## 8. Unified Metrics and Analysis

Values are percentages; parenthesized values are percentage-point deltas from
the accepted baseline.

| Arm | i-AUROC | i-AUPRC | p-AUROC | p-AUPRC | p-AUPRO | Retention | Positive |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| baseline | 97.2100 | 98.7432 | 97.3747 | 58.1160 | 93.8160 | pass | -- |
| density0 | **97.3680 (+0.1579)** | **98.7876 (+0.0444)** | 97.3087 (-0.0661) | 58.4241 (+0.3080) | 93.8218 (+0.0057) | **pass** | **pass** |
| cls_soft_w10 | 97.0286 (-0.1815) | 98.6557 (-0.0874) | 97.3303 (-0.0445) | 58.0419 (-0.0742) | 93.0887 (-0.7273) | fail | fail |
| guided_r8 | 97.1024 (-0.1077) | 98.7666 (+0.0234) | **97.7740 (+0.3993)** | **62.1455 (+4.0295)** | **94.2004 (+0.3844)** | fail | fail |

`density0` is the only arm passing both preregistered gates. Its p-AUPRC gain
is broad enough to be meaningful (10 improved / 5 degraded classes), although
five classes lose more than 0.10 point and `zipper` loses 2.82 points. The
image gain is driven partly by `screw` (+1.33 i-AUROC), while `transistor`
loses 0.21 i-AUROC and 0.72 i-AUPRC. It is therefore a promising simplification
to confirm, not yet a universal setting.

`cls_soft_w10` fails clearly. It loses 0.18 i-AUROC and 0.73 p-AUPRO points;
p-AUPRO degrades in 13/15 classes. The fixed AD2 CLS retrieval weight should
not be carried into AD1.

`guided_r8` creates the strongest localization result: p-AUPRC improves on
all 15 classes, p-AUROC on 14/15, and p-AUPRO on 13/15. However, i-AUROC loses
0.1077 point, narrowly exceeding the 0.10-point gate. The main conflict is
`screw`, with -2.07 i-AUROC and -1.08 i-AUPRC despite +4.47 p-AUPRC; guided
refinement improves spatial localization while changing the top-1% image
ranking unfavorably. Under the locked gate it cannot replace the fixed AD1
baseline, even though it remains useful as a localization diagnostic.

## 9. Continuation Assessment and Conclusion

Verdict:
`KEEP_DENSITY0_DIAGNOSTIC / KILL_CLS_SOFT_W10 / KILL_FIXED_GUIDED_R8_FOR_AD1_CLAIM / BLOCKED_BASELINE`.

For the AD1 branch, remove the density term first if selecting one component
from these tests. The next valid confirmation is not a weight sweep: repeat
`density0` against the original baseline across multiple support selections
or seeds, keeping every other factor fixed. CLS soft retrieval at weight 10
should stop. Fixed guided-r8 should not be advertised as an all-five AD1 win,
but its across-class pixel gains justify later testing with an image score
computed from the unrefined map and a pixel score from the refined map; that
would be a new preregistered arm, not a reinterpretation of this result.

Strict external superiority remains `BLOCKED_BASELINE` because no
same-condition VisionAD/SuperAD artifact is part of this diagnostic.

## Post-Conclusion Storage Cleanup

Only compact metrics, manifests, logs, summary tables, and cleanup evidence
were retained. Remote generation deleted source and refined maps after metric
generation, and the pulled result roots contain zero dense map arrays.
