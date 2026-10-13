# SPCT B1 Full-Support TESTpublic Stage -1 Report

Date: 2026-08-12  
Status: `RUNNING / V2`  
Verdict ceiling: `PROMISING_DIAGNOSTIC`

## Scope

This report tracks the full-support TESTpublic launch for the B1 unconditional
tangent branch. The support pool is the full `train/good` population sealed in
`full_train_good_paths.json`.

## Frozen contract

- MVTec AD2 single-image TESTpublic only.
- DINOv2 L/14 snapshot.
- Resolution `672` for all objects except `sheet_metal` at `448`.
- Support-only custody, RAM-only dense maps, GT opened only after all maps.
- Arms: `B0`, `B1`, `N0`, and `RN_FMLK_full`.
- Primary metric: macro `seg_AUROC_0.05`.
- Secondary metrics: macro `seg_F1`, macro `seg_AUPRO_0.05`, pixel AP, coverage.

## Launch note

The frozen v2 snapshot is running on both preregistered GPU3 queues:

- dsba3: `spct_b1_testpublic_dsba3_v2` (`can -> fabric -> walnuts`)
- dsba5: `spct_b1_testpublic_dsba5_v2`
  (`fruit_jelly -> rice -> sheet_metal -> vial -> wallplugs`)

Both preflights passed exact data-count, manifest-hash, FAISS, RAM, free-VRAM,
and host/container GPU-UUID checks. No results are recorded yet. The benchmark
remains incomparable with SuperAD16 because the support budget is full
`train/good`, not the historical 16-shot context budget.
