# FlowTTE Register Failure Phase A Analysis

Date: 2026-07-07
Verdict: KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC

## Negative Evidence Intake

The preceding structural-register run failed the method claim gate:

| Method | `seg_AUROC_0.05` | `seg_F1` | Delta AUROC vs no-context | Delta F1 vs no-context |
|---|---:|---:|---:|---:|
| no-context DINOv3 | 0.797743 | 0.437800 | 0.000000 | 0.000000 |
| CLS w10 diagnostic | 0.805427 | 0.447118 | +0.007684 | +0.009318 |
| register top-M=4 | 0.798846 | 0.434411 | +0.001103 | -0.003389 |
| register-conditioned NF | 0.796554 | 0.436152 | -0.001189 | -0.001648 |

This analysis is not a hyperparameter tuning run. It decomposes the failure into
context separability, support retrieval replacement, and NF latent distortion.

## Motivation

The motivating question is whether DINOv3 register tokens should remain a
structural part of FlowTTE. Specifically:

- Are register tokens more anomaly-aligned than CLS for image-level context?
- Does register top-M preserve or remove useful patch nearest-neighbor evidence?
- Does register-conditioned NF produce useful density separation, even if the
  current segmentation score does not improve?

## Implementable Design

Target dataset: MVTec AD2 single-image.
Data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
Objects: all eight public objects.
Split: full `test_public/good,bad`.
Backbone: `dinov3_vitl16`.
Support: exact 16-shot DINOv3 no-context support paths from
`flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_notte_dw025_20260707_v1`.
Patch sampling: 128 patches per test image.
Context top-M: 4 support images.

Measured diagnostics:

- context separability: `bad-good` support-context distance delta;
- retrieval quality: nearest support-image retention and distance inflation
  under top-M context routing;
- NF distortion: support latent log-variance change and test bad-good NLL
  separation for unconditional vs register-conditioned NF.

## Evaluation Alignment

This is a diagnostic analysis, not a strict SuperAD-16 claim.

Comparable internal baselines:

- no-context DINOv3 Flow-LatentBank;
- `CLS w10` diagnostic;
- structural `register_topm4`;
- structural `register_condnf`.

Recorded SuperAD-16 and reported SuperADD remain context only because this
analysis uses DINOv3 and the non-paper-aligned support/preprocessing branch.

## Code Modification / Creation

Added:

- `scripts/flow_tte_register_analysis_types.py`
- `scripts/flow_tte_register_analysis_extract.py`
- `scripts/flow_tte_register_analysis_rows.py`
- `scripts/flow_tte_register_analysis_metrics.py`
- `scripts/run_flow_tte_register_failure_analysis.py`
- `scripts/run_flow_tte_register_failure_analysis_remote.sh`

Generated:

- `dinov3_noctx_support_paths.json`
- `context_metrics.tsv`
- `retrieval_metrics.tsv`
- `nf_distortion_metrics.tsv`
- `per_object_analysis.tsv`
- `summary.json`

## Added Code Evaluation

Local verification:

- `python3 -m py_compile ...`: passed
- `uv run ruff check ...`: passed
- `uv run pytest tests/test_flow_tte.py tests/test_flow_tte_context.py tests/test_mvtec_classic_adapter.py -q`: 27 passed

Remote preflight in `hun_fsad_tta_012`:

- `python3 -m py_compile ...`: passed
- support JSON present in `/workspace/fsad_tta/...`

All new Python files are below the 250 pure-LOC threshold:

| File | Pure LOC |
|---|---:|
| `flow_tte_register_analysis_types.py` | 151 |
| `flow_tte_register_analysis_extract.py` | 91 |
| `flow_tte_register_analysis_rows.py` | 129 |
| `flow_tte_register_analysis_metrics.py` | 166 |
| `run_flow_tte_register_failure_analysis.py` | 109 |

## Remote Execution

Container: `hun_fsad_tta_012`.
Host GPUs: `0,1,2`; in-container CUDA slots: `0,1,2`.
Run name: `flowtte_register_failure_phaseA_20260707_v1`.
Remote root:
`/workspace/results_remote/flowtte_register_failure_phaseA_20260707_v1`.
Local pullback:
`results/remote_runs/dsba3/flowtte_register_failure_phaseA_20260707_v1`.

Chunks:

- GPU0: `can,fabric,fruit_jelly`
- GPU1: `rice,vial,wallplugs`
- GPU2: `walnuts,sheet_metal`

After completion, GPUs 0/1/2 returned to idle.

## SuperAD Baseline and Unified Metrics

This run does not produce a new SuperAD comparison metric. It is linked to the
existing segmentation metrics through per-object deltas in
`per_object_analysis.tsv`.

Primary analysis artifacts:

- `context_metrics.tsv`: 24 rows
- `retrieval_metrics.tsv`: 48 rows
- `nf_distortion_metrics.tsv`: 48 rows
- `per_object_analysis.tsv`: 8 rows
- `summary.json`

## Results and Analysis

### Context Separability

Positive context separability means test bad images are farther from support
normal context than test good images.

| Source | Mean `bad-good` min-distance delta | Best-source count |
|---|---:|---:|
| CLS | +0.008771 | 6/8 |
| register | +0.001372 | 2/8 |
| CLS+register | +0.006654 | 0/8 |

CLS is the stronger anomaly-aligned global context source. Register only wins
on `can` and `wallplugs`, and both are weak/near-zero deltas.

### Retrieval Routing

Register top-M is less disruptive than CLS top-M in a nearest-neighbor
retention sense:

| Source/split | Retained nearest support | Distance inflation |
|---|---:|---:|
| CLS good | 0.493914 | 0.783518 |
| CLS bad | 0.491986 | 0.804171 |
| register good | 0.517313 | 0.731858 |
| register bad | 0.522465 | 0.715917 |
| CLS+register good | 0.500885 | 0.762355 |
| CLS+register bad | 0.499408 | 0.785929 |

This explains why `register_topm4` did not catastrophically collapse AUROC.
However, lower retrieval disruption did not translate into better F1 because
register context separability is weak.

### Conditional NF

Register-conditioned NF reduces support latent log-variance on every object:

- mean `logvar_cond - logvar_uncond = -0.016366`

It also improves bad-good NLL separation on 7/8 objects:

- improved: `can`, `fabric`, `fruit_jelly`, `rice`, `vial`, `wallplugs`,
  `sheet_metal`
- worse: `walnuts`

But this density separation does not yet produce a stronger segmentation score.
The current score remains dominated by latent distance, and the structural
`register_condnf` run still lost on mean AUROC/F1.

## Continuation Assessment

Strict method claim now: no.

Register-only routing should stop for claim purposes:

- register context separability is below CLS on 6/8 objects;
- register top-M preserves neighbors better, but this is not enough to improve
  F1.

Register-conditioned NF remains a diagnostic mechanism, not a claim:

- it has a real NLL separation signal on 7/8 objects;
- it compresses the normal latent volume consistently;
- it needs a controlled scoring or map-morphology check before another
  benchmark-scale run.

The single next diagnostic, if continued, should be one of:

- reduced-object map morphology audit for `fabric,can,wallplugs,vial`;
- or one hybrid structural run: CLS for memory routing, register for NF
  conditioning.

Do not continue broad register-only top-M or weight sweeps.

## Conclusion

The Phase A analysis supports the interpretation that DINOv3 register tokens
are not strong direct routing keys for anomaly localization in this setup.
They are better treated as a weak conditioning signal for density modeling.

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

Asset retained:

- reusable Phase A analysis runner;
- evidence that CLS is the stronger context selector;
- evidence that register-conditioned NF changes density geometry but is not yet
  scoring-aligned.

## Post-Conclusion Storage Cleanup

No `anomaly_maps/` were generated by this analysis. Local pullback check found
zero `anomaly_maps/` directories under
`results/remote_runs/dsba3/flowtte_register_failure_phaseA_20260707_v1`.
