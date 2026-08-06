# DSCR Paired Acquisition Interaction Audit

Date: 2026-07-16  
Target: MVTec AD2 single-image public mechanism diagnostic  
Verdict: **KILL_FOR_CLAIM / NO_CONTINUE**

## Stage 0 — Negative Evidence Intake

The preceding DSCR compact cell rejected unconditional high-resolution density
NLL (D0) against the same-memory global cosine 1-NN hard null (G0). PAIA did not
reopen density tuning, calibration, conditional flow, coordinate conditioning,
foreground mixtures, local correspondence, registration, reconstruction, or a
capacity sweep. It reran the already frozen G0/D0 pair only to identify whether
the loss was induced specifically by acquisition shift or was already present in
the matched `regular` condition.

The maximum admissible outcome was `KILL_FOR_CLAIM / PROMISING_DIAGNOSTIC`.
PAIA could preserve an acquisition-specific causal hypothesis, but could never
promote D0 as a method.

## Stage 1 — Frozen Design and Hard Null

- Objects: `fabric`, `rice`, `vial`.
- Seed/fold: seed 0, fold 0.
- Support: P16 split with 12 memory normals and four disjoint held-out normals.
- Queries per object: one good and one bad source identity under each of
  `regular`, `overexposed`, `shift_1`, and `shift_2`.
- Feature geometry: frozen DINOv3 H+ layer 7, native high-resolution tiling.
- G0: unrestricted global cosine 1-NN residual.
- D0: two coupling layers, three epochs, LR `2e-4`, 32,768-token cap.
- Execution: one object per GPU on dsba3 GPUs 0--2, with one retained code
  snapshot and no method-file synchronization.

For metric `M`, object `o`, and condition `c`:

```text
delta_M(o,c) = M(D0 | o,c) - M(G0 | o,c)
interaction_M(o) = mean(delta_M(o,shifted)) - delta_M(o,regular)
shifted = {overexposed, shift_1, shift_2}
```

Float32 TIFF metrics are primary. Float16 metrics are a parity diagnostic.
This three-object, one-source-pair-per-condition cell is `comparable=false` and
does not support a SuperAD/RN-FMLK or population-generalization claim.

## Stage 2 — Evaluator and Verification

Two offline utilities were added:

- `scripts/evaluate_dscr_density_condition_audit.py` enforces exact G0/D0
  condition coverage and ID parity, rejects missing, extra, duplicate,
  nonfinite, constant, or shape-mismatched maps, and computes float32/float16
  pAUROC@0.05, oracle F1, and AP.
- `scripts/aggregate_dscr_density_condition_audit.py` applies the five sealed
  continuation checks without accepting run-time thresholds.

Local verification passed Python compilation, both CLI help checks, diff
checks, and 18 focused plus related regression tests. `ruff` was not installed
as a local Python module, so lint could not be rerun in the leader environment;
the evaluator implementation pass reported a clean Ruff check in its
environment.

All three manifests contain 16 support paths, 12 memory paths, four held-out
paths, and the exact eight preregistered queries. All three completion files
report eight images. Controller logs contain eight `query_complete` events and
no traceback. Before cleanup, every G0/D0/D0c arm contained exactly eight TIFF
maps; D0c was generated only for runner parity and was excluded from PAIA.

The first concurrent `fabric`/`rice` offline evaluation sessions yielded back to
the controller before retaining metrics. Both were rerun sequentially without
regenerating or modifying maps; the sequential evaluations completed
successfully with zero-byte stderr logs.

## Stage 3 — Primary Interaction Results

The following rows are float32 D0-minus-G0 deltas.

| object | regular Δp/ΔF1 | shifted-mean Δp/ΔF1 | interaction Δp/ΔF1 |
|---|---:|---:|---:|
| fabric | -0.008751 / +0.033891 | -0.012587 / +0.015986 | -0.003836 / -0.017905 |
| rice | +0.069926 / +0.023264 | +0.043668 / +0.013066 | -0.026258 / -0.010198 |
| vial | -0.128838 / -0.129513 | -0.119924 / -0.096658 | +0.008914 / +0.032856 |
| **macro** | **-0.022554 / -0.024119** | **-0.029614 / -0.022535** | **-0.007060 / +0.001584** |

The gate required nonnegative macro regular deltas, no regular object below
`-0.03` pAUROC or `-0.05` F1, macro pAUROC interaction at most `-0.03` with at
least two negative objects, negative macro F1 interaction, and complete valid
coverage. Coverage passed; every substantive condition failed:

| gate | result |
|---|---:|
| macro regular pAUROC/F1 nonnegative | fail |
| no regular per-object harm violation | fail (`vial`) |
| pAUROC interaction <= -0.03 and negative on >=2 objects | fail (-0.007060; 2/3) |
| F1 interaction < 0 | fail (+0.001584) |
| complete finite nonconstant float32 coverage | pass |

## Stage 4 — Condition-Level Decomposition

| object | condition | ΔpAUROC | ΔF1 | ΔAP |
|---|---|---:|---:|---:|
| fabric | regular | -0.008751 | +0.033891 | +0.000918 |
| fabric | overexposed | -0.011980 | +0.027237 | -0.007769 |
| fabric | shift_1 | -0.010994 | +0.035519 | -0.004239 |
| fabric | shift_2 | -0.014787 | -0.014798 | -0.029926 |
| rice | regular | +0.069926 | +0.023264 | +0.007567 |
| rice | overexposed | +0.155902 | +0.057452 | +0.019063 |
| rice | shift_1 | -0.030151 | -0.017955 | -0.002386 |
| rice | shift_2 | +0.005254 | -0.000299 | +0.000180 |
| vial | regular | -0.128838 | -0.129513 | -0.090704 |
| vial | overexposed | -0.122040 | -0.104713 | -0.074418 |
| vial | shift_1 | -0.111027 | -0.082383 | -0.053844 |
| vial | shift_2 | -0.126706 | -0.102876 | -0.072536 |

The maximum absolute float32-versus-float16 metric difference was `0.000307`
(`rice`, overexposed, D0 F1), far below any gate margin. Quantization therefore
does not explain the verdict.

## Stage 5 — Deeper Structural Analysis

### 1. The main failure is object-conditional, not shift-induced

At `regular`, D0 already loses macro pAUROC and F1. `vial` alone loses
`0.128838/0.129513`, violating both no-harm floors before any acquisition
change. The shifted mean is only `0.007060` pAUROC worse than this already-bad
regular baseline, while F1 is slightly less harmful. This rejects the narrow
claim that acquisition shift is the principal cause of the density failure.

### 2. A single pooled “shift” variable is too coarse

The response direction depends on both object and shift type. On `rice`,
overexposure greatly amplifies the density advantage (`+0.155902` pAUROC), but
`shift_1` reverses it (`-0.030151`). On `fabric`, pAUROC is mildly negative in
all four conditions, while oracle F1 stays positive until `shift_2`. These are
interactions among object appearance, acquisition mode, and score geometry;
they are not a shared monotone nuisance axis that an unconditional density
head can absorb.

### 3. The positive vial interaction is not robustness

`vial` has a positive difference-in-differences only because D0 is already near
its poor operating regime at `regular`. G0 degrades under shifted acquisition,
but D0 has less remaining performance to lose. D0 remains worse than G0 under
every vial condition by `0.111027` to `0.128838` pAUROC. Treating the positive
interaction as robustness would confuse floor compression with useful shift
invariance.

### 4. Oracle F1 gains do not repair low-FPR ordering

For `fabric`, D0 can raise oracle F1 while lowering pAUROC and usually AP. This
means a threshold can isolate a somewhat better high-score region even though
the low-FPR rank ordering is worse. The effect disappears on `shift_2` and is
not transferable to `vial`. It is score-field reshaping, not a stable ranking
improvement.

### 5. Host and implementation differences are ruled out empirically

The same eight-query pooled compact evaluator reproduced the preceding DSCR
cell. For `fabric` and `rice`, all D0/D0c pAUROC, F1, and AP values matched
exactly, while G0 discrepancies were below `5.5e-7`. The new dsba3 `vial` run
matched the prior dsba5 run within `1.1e-6` across G0/D0/D0c pooled pAUROC, F1,
and AP. The earlier severe vial loss is therefore not a dsba5 GPU or code-hash
artifact.

## Stage 6 — Unified Pooled Reproduction

The pooled float16 rows below use the same eight query maps as the preceding
DSCR cell and are included only as a reproduction audit.

| object | G0 pAUROC/F1/AP | D0 pAUROC/F1/AP | D0-G0 pAUROC/F1/AP |
|---|---:|---:|---:|
| fabric | 0.527093 / 0.216515 / 0.156934 | 0.514495 / 0.242294 / 0.151726 | -0.012597 / +0.025779 / -0.005208 |
| rice | 0.600696 / 0.020435 / 0.002653 | 0.650914 / 0.042351 / 0.007771 | +0.050218 / +0.021916 / +0.005118 |
| vial | 0.673175 / 0.156020 / 0.088699 | 0.551273 / 0.052433 / 0.018979 | -0.121902 / -0.103587 / -0.069720 |

## Stage 7 — Decision and Method Mapping

PAIA closes the acquisition-conditioning explanation for this density family.
The result does not justify conditional flow, mode routing, whitening,
Mahalanobis scoring, calibration, fusion, threshold tuning, or another density
capacity sweep. Those paths either already have negative evidence or require a
regular-condition scorer that this branch does not provide.

The distribution-shift problem remains central, but the evidence changes the
required solution shape: a future candidate must first preserve defect evidence
under `regular` for every object, then demonstrate that its nuisance estimate is
shared across acquisition changes without collapsing object-specific anomaly
geometry. D0 fails the first requirement, so adding a shift conditioner to D0
would condition a structurally misaligned score rather than repair it.

Because the cell contains one good and one bad source identity per condition,
the heterogeneous condition effects are mechanism evidence only. No confidence
interval, pixel-i.i.d. significance claim, or all-object generalization claim is
made.

## Post-Conclusion Storage Cleanup

Compact manifests, completion files, logs, per-object condition and pooled
metrics, and the aggregate summary are retained under `remote/` and
`summary.json`. After metric and report verification, all PAIA `anomaly_maps/`
directories were deleted from the three dsba3 output roots. The final remote
audit returned zero map directories and zero TIFF files for `fabric`, `rice`,
and `vial`; no dense maps were pulled locally.
