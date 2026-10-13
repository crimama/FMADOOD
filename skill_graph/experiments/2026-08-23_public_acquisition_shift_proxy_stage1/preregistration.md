# Public Acquisition-Shift Proxy Stage 1 Preregistration

Date: 2026-08-23
Protocol: `public-acquisition-shift-proxy-stage1-20260823`
Status: **FROZEN before any code/GPU execution** under `docs/EXPERIMENT_GUIDELINES.md`.
Claim class: protocol-validation diagnostic. `benchmark_comparable=false` until gates pass;
this experiment is the guideline G-5 calibration for acquisition-shift proxies.
Compute: local RTX 4090 only. Estimated cost: Baseline1-core map regeneration ~15-25 GPU-min
(observed `60-100 s/object` in Stage SAM-A), SuperAD-16 ~1-2 GPU-h, stratified evaluation
CPU-bound ~30-60 min. Failure still yields the stratified robustness table as paper content,
so the runway check passes.

## 1. Research question

The official-server submission showed Baseline1-core degrades least from TESTprivate to
TESTprivate-mixed (Seg-F1 `-4.11%`, versus SuperAD-16 reported `-9.90%`; AU-PRO `-2.32%`
versus RoBiS reported `-11.30%`). This is a two-point observation that cannot be repeated
against the rate-limited server and currently cannot enter the paper as a contribution.

Question: does stratifying the PUBLIC split by its acquisition-condition tokens
(`regular / overexposed / underexposed / shift_1..3`) reproduce the known server-side
robustness ordering between the two methods we can rerun locally? If yes, the stratified
public protocol becomes a repeatable shift-robustness measurement for any method.

## 2. Why this can validate anything

MVTec AD2 public conditions are real acquisition shifts (not synthetic transforms), so this
doubles as the guideline G-5 calibration: if real-condition stratification tracks the server's
mixed-condition behavior for both available methods, future Stage -1 diagnostics may use
condition-stratified public evaluation as their external-validity lane.

Known server anchors (frozen, from `PaperWorks/Main_Table_Baseline_Comparison.md`):
- Baseline1-core: priv Seg-F1 `0.4594` -> mixed `0.4405` (`-4.11%`).
- SuperAD-16 (reported): `0.4718` -> `0.4251` (`-9.90%`). No server AU-PRO value exists for
  SuperAD-16, so AU-PRO ordering is descriptive only.

## 3. Arms and custody

| Arm | Config source | Notes |
|---|---|---|
| `baseline1_core` | exact submission arguments (`scripts/run_baseline1_private_submission.py` assertions; identical CLI to SAM Stage-1 `generate_basemaps.baseline_args`) | DINOv3-H+/16, P16 fixed manifest, density-weight 0 |
| `superad16` | `configs/baselines/same16_superad_ad2.yaml` (Same16 v6 frozen contract) | DINOv2-L/14, same SuperAD-selected support paths, seed 0 |

Both rerun locally on all eight objects, full `test_public/good,bad`. Condition tokens are
read from filenames AFTER map generation, post-scoring only (established allowed practice;
never an inference input). Static inference throughout (R7): no query-time update, no TTA.

## 4. Conditions and families (frozen)

- Cells: `{good,bad} x {regular, overexposed, underexposed, shift_1, shift_2, shift_3}`.
- `shift_4` is excluded a priori (20 images globally; no stable cell statistics).
- Families: `exposure = {overexposed, underexposed}`; `shiftfamily = {shift_1, shift_2, shift_3}`.
- Primary family: `exposure` (the private-mixed condition is exposure-mixing; pre-declared).
  `shiftfamily` is a pre-registered secondary descriptive endpoint. No other grouping may be
  selected after seeing results.
- Known gaps handled a priori: `fruit_jelly` lacks `shift_2/shift_3`; family means skip absent
  cells and record the skip.

## 5. Metrics and threshold policy (G-4)

Per `object x method x cell`: pAUROC@0.05 (`seg_AUROC_0.05`), AU-PRO@0.05, and
Seg-F1 at the object-level pooled-public-oracle threshold (provenance-labeled
`calibrated/public-oracle-transferred`):
- Baseline1-core thresholds: the exact frozen floats used for the official submission
  (`configs/baselines/baseline1_core_public_thresholds.json`) — byte-level custody with the
  server protocol.
- SuperAD-16: its own per-object pooled-public-oracle best-F1 threshold computed from its
  regenerated maps (method-level analog, computed before any condition split is scored).
Macro = mean over eight objects per condition. Degradation endpoint:
`Delta_c = M(condition c) - M(regular)` per method (macro).

## 6. Gates (frozen)

- **V1a reproduction (Baseline1-core)**: pooled all-condition oracle Seg-F1 reproduces
  `0.530177` within `1e-3`; pAUROC@0.05 reproduces `0.839561` within `1e-3`
  (recorded anchors, `results/baseline1_knn5_default_ab_20260807_v5/summary.json`).
- **V1b reproduction (SuperAD-16)**: pooled pAUROC/F1/AU-PRO reproduce Same16 v6 means
  `0.761857/0.368423/0.534924` within `5e-3` (cross-environment tolerance).
- **V2 ordering (PRIMARY)**: exposure-family Seg-F1 degradation is strictly smaller for
  Baseline1-core than SuperAD-16: `Delta_exposure(B1C) > Delta_exposure(SuperAD-16)`
  (server direction: `-4.11%` vs `-9.90%`).
- **V3 validity**: every gated cell has >= 4 good and >= 4 bad images (except the declared
  `fruit_jelly` gap); all maps finite; duplicated-condition parity spot-check on one object
  rerun end-to-end with max abs delta <= `1e-3` on every cell metric.
- Descriptive (no gate): AU-PRO/pAUROC degradation curves per family; `shiftfamily` results.

Interpretation contract: V1a+V1b+V2(+V3) pass -> proxy `VALIDATED_ONE_PAIR`; the stratified
protocol is promoted to the paper's robustness section with the single-pair validation stated
explicitly. Any failure -> `KILL_FOR_CLAIM` for the proxy; the server observation remains a
reported anecdote. No post-hoc family/threshold changes; G-2 iteration lane: none declared.

## 7. Boundaries

No TESTprivate/mixed access (server-reported numbers are quoted context only). No tuning of
either arm beyond its recorded frozen configuration. Condition tokens never enter scoring.
Dense maps retained until aggregation completes, then removed after compact artifacts +
hashes are secured.
