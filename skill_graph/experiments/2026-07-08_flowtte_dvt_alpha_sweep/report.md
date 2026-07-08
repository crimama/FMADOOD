# FlowTTE DVT Alpha Sweep Diagnostic

## 1. Motivation

The previous DVT position denoise run used `alpha=1.0` and improved the all-object mean over no-DVT, but the object pattern was uneven. This follow-up tests whether the gain is a stable monotonic effect or an object-dependent artifact-removal strength issue.

Negative-evidence intake: this is not a new method claim. It is a bounded diagnostic inside the existing Flow-LatentBank no-TTE branch. The likely failure basin is label/oracle-only alpha selection or passive feature surgery. Therefore the sweep is only used to decide whether a support-only alpha-selection rule is worth implementing.

## 2. Implementable Design

- Dataset: MVTec AD2 single-image.
- Remote data root: `/home/hunim/Volume/DATA/mvtec_ad_2`.
- Objects: `can, fabric, fruit_jelly, rice, vial, wallplugs, walnuts, sheet_metal`.
- Method: Flow-LatentBank no-TTE with DINOv3 features and DVT-style support position-mean denoise.
- Denoiser: `G_pos = mean_support_feature_at_position - global_support_feature_mean`; `p_clean = p_raw - alpha * G_pos`.
- Alpha grid: `0.00` no-DVT baseline, `0.25`, `0.50`, `0.75`, `1.00`.
- Primary metrics: `seg_AUROC_0.05`, `seg_F1`.
- Strict claim gate: not applicable here because alpha selection uses post-hoc metric inspection and SuperAD/RN-FMLK hard-null gates are not re-run in this report.
- Continuation gate: continue only if at least one alpha improves mean over no-DVT and the object-level oracle indicates a plausible support-only gate could avoid the damaged classes.

## 3. Evaluation Alignment

The candidate and no-DVT baseline share the same MVTec AD2 object set and segmentation evaluator. The alpha sweep is internally comparable to the previous Flow-LatentBank no-TTE baseline, but it is not a same-condition SuperAD claim because alpha selection is diagnostic and post-hoc.

## 4. Code Modification / Creation

Created launcher:

- `scripts/run_flow_tte_dvt_alpha_sweep_remote.sh`

The launcher wraps the existing all-8-object DVT runner and skips completed alpha roots when `remote_run_complete.txt` exists.

## 5. Added Code Evaluation

- `bash -n scripts/run_flow_tte_dvt_alpha_sweep_remote.sh` passed locally.
- Remote copied launcher also passed `bash -n` before execution.

## 6. Remote Experiment Execution

Executed on dsba3 inside container `hun_fsad_tta_012` using host GPUs `0,1,2`:

```bash
cd /workspace/fsad_tta && ALPHAS="0.25 0.5 0.75" RUN_SUFFIX=20260708_v1 bash scripts/run_flow_tte_dvt_alpha_sweep_remote.sh
```

Pulled local result roots:

- `results/remote_runs/dsba3/flowtte_dvt_denoising_all8_a025_20260708_v1`
- `results/remote_runs/dsba3/flowtte_dvt_denoising_all8_a05_20260708_v1`
- `results/remote_runs/dsba3/flowtte_dvt_denoising_all8_a075_20260708_v1`

Previous comparison roots:

- No-DVT baseline: `results/remote_runs/dsba3/flow_latentbank_mvtecad2_all8_shot16_dinov3vitl16_notte_dw025_20260707_v1`
- DVT alpha 1.0: `results/remote_runs/dsba3/flowtte_dvt_denoising_all8_a10_20260707_v1`

## 7. Results and Analysis

### Alpha Mean Summary

| alpha | mean AUROC_0.05 | mean F1 | delta AUROC vs no-DVT | delta F1 vs no-DVT | AUROC wins | F1 wins |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.797743 | 0.437800 | 0.000000 | 0.000000 | 0/8 | 0/8 |
| 0.25 | 0.800444 | 0.442082 | 0.002701 | 0.004282 | 5/8 | 3/8 |
| 0.50 | 0.805892 | 0.446748 | 0.008149 | 0.008948 | 5/8 | 4/8 |
| 0.75 | 0.821793 | 0.459839 | 0.024049 | 0.022039 | 4/8 | 4/8 |
| 1.00 | 0.825207 | 0.468348 | 0.027464 | 0.030548 | 4/8 | 3/8 |


Best mean AUROC alpha: `1.00` with `0.825207`. Best mean F1 alpha: `1.00` with `0.468348`.

### SuperAD / SuperADD Context

Recorded SuperAD-16 in the current component table is `0.765802` AUROC_0.05 / `0.385534` F1. The fixed alpha `1.00` DVT result is above that context by `+0.059405` AUROC and `+0.082814` F1.

Reported SuperADD TESTpublic context is `0.839300` AUROC_0.05 / `0.626113` F1. The fixed alpha `1.00` DVT result remains below that by `-0.014093` AUROC and `-0.157764` F1. Even the oracle alpha diagnostic remains below SuperADD in F1, so this branch is not claim-ready.

### Oracle Diagnostic Upper Bound

- Oracle-by-AUROC mean AUROC_0.05: `0.833359` (`+0.035615` vs no-DVT).
- Oracle-by-AUROC mean F1 at selected alpha: `0.474177`.
- Oracle-by-F1 mean F1: `0.476974` (`+0.039174` vs no-DVT).
- Oracle-by-F1 mean AUROC at selected alpha: `0.827210`.

Oracle choices are diagnostic only because they use ground-truth metrics after evaluation. They show the maximum value of a future no-label support-only selector, not a valid deployable policy.

### Object-Level Pattern

See `per_object_alpha.tsv`, `oracle_by_auroc.tsv`, and `oracle_by_f1.tsv` for full values. The main pattern is asymmetric: some objects benefit from strong position-artifact subtraction, while others are harmed by the same alpha. That means fixed `alpha=1.0` is not yet a clean method component even though it improves mean AUROC and mean F1 over no-DVT.

## 8. Conclusion

Verdict: `KILL_FOR_CLAIM / CONTINUE_DIAGNOSTIC`.

The fixed-alpha DVT denoiser cannot be claimed as a robust method component yet because it has object-level no-harm failures and the best alpha is selected diagnostically. However, the mean improvement and oracle upper bound justify one constrained continuation: implement a support-only alpha selector using support feature compactness or held-out support self-consistency, then test whether it recovers the oracle direction without labels.

Hard-stop condition for the next step: if a support-only selector cannot beat fixed alpha `1.0` or no-DVT on both mean AUROC_0.05 and mean F1 without increasing object-level harm, stop the DVT-position-denoise branch.

## 9. Storage Cleanup

All new remote runs were launched through the cleanup-enabled DVT runner. Pulled chunk roots include `cleanup_evidence.txt`.

Final verification:

- Local pullback roots for alpha `0.25`, `0.50`, and `0.75` contain `remote_run_complete.txt`.
- Local pullback roots contain `0` directories named `anomaly_maps`.
- Remote roots under `/home/hunim/Volume/FMAD-OOD-remote/results_remote/` contain `remote_run_complete.txt`.
- Remote roots contain `0` directories named `anomaly_maps`.
- dsba3 GPU status after completion: GPUs `0,1,2,3` each reported `1 MiB` used and `0%` utilization.

## Artifacts

- `alpha_summary.tsv`
- `per_object_alpha.tsv`
- `oracle_by_auroc.tsv`
- `oracle_by_f1.tsv`
- `summary.json`
