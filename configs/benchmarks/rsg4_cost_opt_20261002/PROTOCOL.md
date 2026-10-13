# RSG4 cap1024 cost optimization — development experiment

Date: 2026-10-02. Authorized resources: dsba3 physical GPUs 0, 1, 2, existing container.

## Hypothesis and boundary

Retaining GPU features and reusing leave-source-out calibration banks reduces fitting overhead. Computing shared token distances once across bags can reduce query search work without changing per-bag minima. Backbone-only FP16/BF16 autocast may reduce the dominant feature extraction cost; these are numerical precision ablations and require accuracy validation. No backbone parameters are trained. Matching, calibration and gating remain FP32.

All arms retain cap1024, four bags, layers [5,11,23], resolutions [448,672,896,1120], normal-only support, preprocessing, noise guard and component quantile postprocessing. The baseline is canonical reference_matching FP32. fast_exact changes transfer/calibration implementation; fast_shared adds shared bag distances and chunk512; fast_fp16 and fast_bf16 add the declared backbone precision change. Support is refitted for every arm.

## Stage 1: bounded speed and numerical screen

- GPU0: MVTec AD 2 fabric, 16-shot; GPU1: MVTec AD 2 sheet_metal, 16-shot; GPU2: VisA pcb1, 1-shot.
- Seed 0, existing fixed manifests named in each YAML. Five arms per class, run sequentially on the same physical GPU. Other GPUs are outside the allocation.
- Twelve fixed queries/class: six normal and six anomalous, deterministically spread over sorted paths. Labels are used only for this evaluation sampling and later accuracy evaluation, never fitting or prediction. Existing test_public/test populations are development/post-hoc data, not untouched final evaluation.
- Synchronized component profiler reports fit seconds, query mean/p95 seconds, fit/query peak allocated GiB, component absolute seconds and fractions, support/config hashes, calibration scale differences and map differences. Baseline maps are held only in host RAM for paired comparison; no maps or weights are saved.
- Instrumented timing includes synchronization overhead. End-to-end pipeline timing in Stage 2 is the runtime confirmation. Report per-class results, including the expensive sheet_metal class, without silently discarding outliers.
- Stop an arm on nonfinite output, identity mismatch, OOM or exception; record execution failure separately from performance. Do not increase the resource allocation. Investigate FP32 deviations before treating the optimization as equivalent.

## Stage 2: matched full-query development confirmation

Select a finite candidate with useful speed gains and acceptable numerical stability. Use scripts/experiments.py run with baseline and candidate on matching support/query/class scopes, workers=1 per GPU, cost_profile enabled. Include MVTec AD and VisA accuracy checks and AD2. Primary quality metric: AUPRO@0.05 (higher is better); report pAUROC@0.05 and oracle F1 as secondary diagnostics. A provisional macro decrease <=0.2 percentage points is the development acceptance threshold; investigate any class decrease >1 pp. This small development scope cannot establish full-dataset statistical equivalence.

Stage 2 class scope declared before results: AD2 fabric (16-shot), MVTec AD cable (1-shot), VisA pcb1 (1-shot).

Scheduling amendment (2026-10-02, before cable launch): cable may use GPU2 after the VisA suite if GPU1 is still occupied by the sheet_metal screen. All cable arms remain sequential on one physical GPU, selected once before launch and recorded by UUID. This changes scheduling only; support/query/configs and the candidate budget remain fixed.

Budget: 15 initial class/arm profiles; then at most two candidate arms plus the baseline on each of these three classes. Also measure the repository's PatchCore implementation on fabric with FP32 and backbone FP16, identical support/backbone layers/evaluator, to answer the earlier competitor-cost question. PatchCore uses its existing single-scale resolution672 method; RSG4 uses four scales. This is a declared method configuration difference. FP16 is applied to both methods for the optimized precision comparison; PatchCore retains its required NumPy feature interface. This does not reproduce the original PatchCore paper's different backbone.

If no candidate passes the numerical screen, diagnose rather than claiming preserved performance. The existing common runner/collector registers full-query metrics; sampled component profiles remain a separate descriptive report. Preserve remote logs and pull compact metrics/config/manifests.
