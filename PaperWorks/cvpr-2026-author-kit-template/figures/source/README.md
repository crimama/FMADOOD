# Review figure sources

Run from the paper directory using the project's existing Python environment:

    MPLCONFIGDIR=/tmp/scopead-mpl ../../.venv/bin/python figures/source/draw_overview.py
    MPLCONFIGDIR=/tmp/scopead-mpl ../../.venv/bin/python figures/source/draw_review_figures.py
    make all

Outputs are PDF, PNG and SVG files under this paper's figures directory.

- Figure 1 shows real MVTec AD 2 walnuts images: supports train/good/129_regular
  and 266_regular (from configs/methods/VOID_support_16.json), queries
  test_public/good/000_overexposed and test_public/bad/004_overexposed (GT
  contour for display). The distributions are schematic, not measured data. Normal scores are gray, defects red; the dashed line is a schematic
  low-FPR threshold.
- Figure 2 (draw_overview.py) reads
  PaperWorks/figures/cache_rsg4_mechanism/overview_walnuts__defect.npz.
  Panel (d) follows one real defect pixel (row 380, col 133); output is before CQ99.
- Figure 3 uses calibration_quantiles.json, copied from
  results/remote_runs/dsba3/scopead_calibration_diagnostic_20261015/diagnostics/.
  Values are test-normal query Q95 divided by each stage's minimum (display only).
- Figure 4 (resolution) uses RESOLUTION_AUPRO in draw_review_figures.py: single
  resolutions from results/remote_runs/dsba3/scarf_single_resolution_fp16_20261012/
  comparison.md, except 16-shot 1120 from scopead_final_cost_20261019 (same
  configuration, identical to Table 4); 4-res Renyi from scopead_renyi_cq_20261018.
  The dashed line is the best single resolution.
- Figure 5 uses order_rank_diagnostic.json (seed 0, 8-class macro); labels were
  used only after prediction. The dashed line is the uniform 25% share.
- Figure 6 uses lighting_strata.json: official per-condition AUPRO@0.05
  (condition_strata in run_manifest.json, full resolution, CQ99, seed 0).

Palette: baseline/before = #ACB6C2, ours/after = #28638A; orange is reserved
for the shallow layer in Figure 2.

extract_qualitative_maps.py (GPU) produced
PaperWorks/figures/cache_scopead_qualitative/*.npz (256-px CQ99 maps of four
arms). That cache is not used by any manuscript figure: its downsampled AUPRO
does not match the official evaluator.

No experiments are run by these generators.

The added appendix aggregation table is the existing CQ-off comparison in
PaperWorks/outlines/Experiments.md, Table 5B. Final CQ-on layer-weight
results are class macros from the eight rows per summary.tsv under
results/remote_runs/dsba3/scopead_final_validation_20261019/lambda_sweep/.
The default weight uses the final seed-0 results, not a new run.
