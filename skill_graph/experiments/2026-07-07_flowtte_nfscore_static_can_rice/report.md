# FlowTTE-NFScore Static NLL Can/Rice Diagnostic

Date: 2026-07-07
Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_STATIC_NLL`

## Negative Evidence Intake

This branch is not a pure threshold retune of a killed method. It changes the
scoring source from latent memory distance to NF density/NLL. The likely failure
basin is density miscalibration: raw NLL can rank object/background variation
above true defects and collapse best-threshold F1.

## Motivation

The current FlowTTE prototype uses NF as a projection/gating module and scores
mainly by latent-memory distance. The vNext method direction is to use NF as the
scoring module, then later add conservative test-time adaptation. Before adding
TTA, this run tests the smallest static mechanism: raw patch NLL as anomaly map.

## Implementable Design

- Method: `FlowTTE-NFScore` static NLL
- Dataset: MVTec AD2 single-image
- Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`
- Objects: `can`, `rice`
- Split: full `test_public/good,bad`
- Shots: `4`
- Support policy: `first`
- Backbone: `dinov2_vitl14`
- Feature fusion: `layer_norm_mean`
- Flow: 2 coupling layers, hidden multiplier 1, 3 epochs
- Score mode: `nf_nll`
- Expansion budget: `1.0`, so no memory expansion capacity
- Primary diagnostic metrics: `seg_AUROC_0.05`, `seg_F1`
- Immediate baseline: previous `FlowTTE-LatentBank` can/rice 4-shot run

Strict SuperAD claim is out of scope because this is a reduced 4-shot
diagnostic. Recorded SuperAD-16 can/rice values are included as context only.

## Evaluation Alignment

The run tests the intended first gate: whether static NF-NLL is viable before
adding test-time adaptation. The comparable method-vs-method diagnostic is the
previous reduced `FlowTTE-LatentBank` run on the same objects and split.

SuperAD is not same-condition comparable here because the available recorded
artifact is the 16-shot paper-aligned reference, while this diagnostic uses
4-shot support.

## Code Modification / Creation

Modified:

- `src/flow_tte/config.py`: added `ScoreConfig.score_mode`.
- `src/flow_tte/scoring.py`: added `nf_nll` scoring branch using raw NLL.
- `src/flow_tte/pipeline.py`: passes raw NLL to scoring.
- `scripts/flow_tte_mvtec_ad2_core.py`: wires `score_mode` into `ScoreConfig`.
- `scripts/run_flow_tte_mvtec_ad2.py`: exposes `--score-mode`.
- `tests/test_flow_tte.py`: adds a static NF-NLL behavior test.

Created:

- `results/remote_runs/dsba3/flow_tte_nfscore_mvtecad2_can_rice_shot4_static_nll_20260707_v1/comparison_reduced`

## Added Code Evaluation

Local checks:

- `python3 -m py_compile ...`: passed
- `uv run --extra dev ruff check ...`: passed
- `uv run --extra dev basedpyright`: `0 errors`
- `uv run --extra dev pytest -q`: `20 passed`

Remote sanity:

- Synced changed files to `/workspace/fsad_tta`.
- Remote `py_compile` passed.
- Remote CLI help exposes `--score-mode {latent_distance,nf_nll}`.

## Remote Execution

Remote:

- Server/container: dsba3, `hun_fsad_tta`
- Host GPU: `3`; in-container CUDA slot: `0`
- Run id:
  `flow_tte_nfscore_mvtecad2_can_rice_shot4_static_nll_20260707_v1`
- Remote path:
  `/workspace/results_remote/flow_tte_nfscore_mvtecad2_can_rice_shot4_static_nll_20260707_v1`
- Local path:
  `results/remote_runs/dsba3/flow_tte_nfscore_mvtecad2_can_rice_shot4_static_nll_20260707_v1`

## SuperAD Baseline and Unified Metrics

Same-condition SuperAD baseline: `BLOCKED_BASELINE` for this 4-shot reduced
diagnostic.

Context-only SuperAD artifact:
`../FMAD-OOD/configs/baselines/recorded_superad16_mvtec_ad2_8object_metrics.json`

Unified comparison against the immediate FlowTTE baseline:

| Method | mean AUROC@0.05 | mean F1 |
| --- | ---: | ---: |
| FlowTTE-NFScore static NLL | 0.674336 | 0.127093 |
| FlowTTE-LatentBank can/rice 4-shot | 0.765772 | 0.315589 |
| Delta vs LatentBank | -0.091436 | -0.188495 |
| Recorded SuperAD-16 context | 0.757674 | 0.333710 |
| Delta vs SuperAD-16 context | -0.083338 | -0.206616 |

## Results and Analysis

| Object | NFScore AUROC | LatentBank AUROC | Delta | NFScore F1 | LatentBank F1 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| can | 0.549017 | 0.630660 | -0.081643 | 0.000480 | 0.003423 | -0.002943 |
| rice | 0.799655 | 0.900885 | -0.101230 | 0.253706 | 0.627754 | -0.374048 |

Raw NLL scoring is worse on both objects. `rice` is the decisive failure:
AUROC drops by about 10 points and F1 drops by about 37 points. The very high
best thresholds (`1552`, `1624`) also indicate score-scale/calibration mismatch
rather than a clean localization boundary.

## Continuation Assessment

Strict method claim: no.

Continuation for this exact static raw-NLL branch: no. It fails the first gate
before any full AD2 sweep or adapter-only TTA should be attempted.

A different small branch may still be justified, but it should not be raw NLL
alone. The next viable diagnostic would need calibrated NLL, support-relative
NLL z-scoring, or adapter-only TTA with an explicit anchor and a latent-distance
control.

## Conclusion

Verdict: `KILL_FOR_CLAIM / NO_CONTINUE_STATIC_NLL`.

The code utility remains useful: FlowTTE now has a config-driven `score_mode`
that can support future calibrated NF scoring and TTA experiments. The raw
static NLL score itself should not be scaled up.

## Cleanup

Remote and local `anomaly_maps/` trees were removed.

Evidence:

- Local `cleanup_evidence.txt`: `cleanup_anomaly_maps=true`
- Remote `cleanup_evidence.txt`: `cleanup_anomaly_maps=true`
- Remote run directory size after cleanup: `24K`
