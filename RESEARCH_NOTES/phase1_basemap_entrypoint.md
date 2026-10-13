# Phase 1 — Baseline1-core base-map entrypoint (recon by orchestrator, 2026-08-21)

## Runner
- Entry: `scripts/run_flow_tte_mvtec_ad2.py` (validated wrapper: `scripts/run_baseline1_private_submission.py`).
- Baseline1-core config (from `run_baseline1_private_submission.py` assertions):
  - `backbone_model = dinov3_vith16plus`, `feature_layers = (7,15,23,31)`, `coupling_layers = 2`.
  - `dvt_denoise_mode = position_mean`, `dvt_denoise_alpha = 1.0`.
  - `score_field_calibration_mode = none`. `shots = 16` (P16, fixed SuperAD manifest).
  - **`--density-weight 0`** for Baseline1-**core** (raw latent 1-NN geometry).
  - resolution from object info (672; sheet_metal 448).

## Continuous maps (the base maps M for SAM refinement)
- Written to `<output_root>/anomaly_maps/<object>/{good,bad}/*.tiff` (float16 patch maps upsampled to image size, `run_flow_tte_mvtec_ad2.py:557,243`).
- Held-normal maps: `<output_root>/calibration_maps/<object>/good/*.tiff` (`:635`).
- **Do NOT pass `--cleanup-maps`** → maps are retained (default writes a cleanup record only when requested, `:847-852`). This is how we persist base maps for the refinement arms.

## q_prompt source (§4 of prereg)
- §4: `q_prompt` = 99.5th percentile of Baseline1-core map scores over the 16 support LOO self-scores, per object. Practical source: support/held-normal map score distribution; `calibration_maps/<object>/good/*.tiff` is the available held-normal proxy. Finalize exact source in the runner delegation (must be label-free, static, per-object).

## Next-delegation plan (base-map + arms + eval + aggregator)
1. Run the runner for Phase-1 objects `sheet_metal,rice,can,fabric`, Baseline1-core, no cleanup → dump anomaly_maps + calibration_maps.
2. Compute per-object `q_prompt` (label-free) from held-normal map scores.
3. Apply arms B0/M1/N0/N1/N2 (from `scripts/sam_region_refine.py`) to each `test_public` base map.
4. Evaluate AU-PRO@0.05 + `gt_component_recall_at_global_oracle` + pixel-AP + Seg-F1 via `src/post_eval.py` / `src/flow_tte/aupro.py`; run the anomaly-preservation assay (defect-pixel retention inside masks) — GT opened ONLY after all arm maps written+hashed.
5. Aggregate + apply §6 gates; emit custody/access ledger.

## Risks
- Runtime: full DINOv3-H+ map generation for 4 objects on one RTX 4090 is the heavy step; batch/queue sequentially. SAM adds ~1s/image.
- Keep base-map generation and SAM refinement as SEPARATE stages so a rerun of arms doesn't require regenerating base maps.
