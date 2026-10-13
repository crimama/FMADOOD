# TCCR + DB-FSPC Stage -1 Preregistration

Date: 2026-08-11

## Resource Boundary

`dsba3` is excluded from all preflight, sync, and execution for this run.

Frozen allocations:

- local GPU 0 (RTX 4090): TCCR on `can,fabric,vial`
- dsba5 host GPU 3 exposed as container CUDA slot 0 (TITAN RTX): TCCR parity
  on `rice` plus duplicate `can`
- dsba5 host GPU 3 exposed as container CUDA slot 0: DB-FSPC Gate A after the
  remote TCCR shard completes

## TCCR Stage -1

Scope: defect-side cross-image-calibrated region completion, not a claim-tier
result.

Frozen object and shard allocation:

- local shard: `can,fabric,vial`
- remote parity shard: `rice` plus duplicate `can`

Protocol constraints:

- fixed support-16 manifest:
  `skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json`
- deterministic `16 good + 32 bad` public hash per object
- frozen DINOv3 H+/16, layer 7
- required `P16_H+DVT_anchor`
- no query-time adaptation, no private split access
- dense score maps are staged in memory only
- GT masks are opened only after all implemented arm maps have been staged
- critic gates remain exact and frozen; no arm, threshold, or continuation rule
  may be retuned post hoc

Implemented arms:
`P16_H+DVT_anchor`, `raw_g0_layer7`, `raw_tiled_fine_knn`, `lift_only`,
`calibrated_coarse_fine_max`, `calibrated_coarse_fine_mean`,
`matched_coarse_to_fine_predictor`, `one_phase_excess`, `two_phase_tccr`,
`geometry_shuffle`, `phase_shuffle`, `rgb_edge`,
`equal_compute_duplicate_tiles`. The raw tiled comparator observes both frozen
phases; the duplicate control independently repeats phase 0 at equal forward
compute and is not query-to-query matching.

Primary diagnostic readouts: support LOO tail coverage, beta, component seed
top-0.5% recall, AU-PRO, pAUROC, and fixed/oracle F1 family metrics.

Continuation is allowed only if the preregistered two-phase TCCR improves seed
recoverability and AU-PRO/component recall over raw G0 and raw tiled fine
without being explained by geometry/phase/null arms, predictor nulls, or the
required P16 H+DVT anchor contract.

## DB-FSPC Gate A

Scope: dsba5-only label-free support-normal structural feasibility audit. It
does not read test/public/private/GT data and cannot emit anomaly scores.

Frozen runtime contract:

- model: `Qwen/Qwen3-VL-2B-Instruct`
- model revision: `8964489`
- runtime: `vLLM 0.11.0`
- transformers: `4.57.1`
- qwen-vl-utils: `0.0.14`
- numpy: `1.26.4`
- isolated environment on dsba5

Gate A objects: `can,vial,fabric,rice`.
Frozen sentinels: `[0, 5, 10, 15]`.

Gate A measures JSON success, vocabulary Jaccard, count agreement, and box
repeatability across three defect-free prompt paraphrases and low-resolution
support perturbations. Gate pass requires JSON >= 0.99, vocabulary >= 0.80,
count agreement >= 0.90, box repeatability >= 0.60.

Gate A can run only after the remote TCCR shard completes. It cannot license
Gate B unless TCCR earns `PROMISING_DIAGNOSTIC`.
