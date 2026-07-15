#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/home/woojun/dataset/mvtec_ad}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flow_latentbank_mvtecad1_density0_guided_hybrid_s1_2_4_8_20260713_v1}"
RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
OBJECTS="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"

cd "${FSAD_ROOT}"
mkdir -p "${RUN_ROOT}/logs"
export FMAD_DINOV2_OFFLINE="${FMAD_DINOV2_OFFLINE:-1}"

run_shot() {
  local gpu="$1"
  local shot="$2"
  local shot_root="${RUN_ROOT}/shot_${shot}"
  local raw_root="${shot_root}/density0_raw"
  local guided_root="${shot_root}/guided_r8"
  local hybrid_metrics="${shot_root}/hybrid_metrics.json"
  local log="${RUN_ROOT}/logs/shot_${shot}.log"
  if [[ -f "${hybrid_metrics}" && -f "${guided_root}/run_manifest.json" ]]; then
    echo "[skip] shot=${shot}" | tee -a "${log}"
    return
  fi

  echo "[start] shot=${shot} gpu=${gpu}" | tee "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/run_flow_tte_mvtec_ad1.py \
    --data-root "${DATA_ROOT}" \
    --output-root "${raw_root}" \
    --project-root /workspace \
    --fsad-root "${FSAD_ROOT}" \
    --objects "${OBJECTS}" \
    --shots "${shot}" \
    --seed 0 \
    --device cuda \
    --backbone-model dinov2_vitb14_reg \
    --preprocess-recipe fmad_shorter_edge \
    --image-size 448 \
    --crop-size 448 \
    --feature-layers 2,5,8,11 \
    --feature-fusion layer_norm_mean \
    --support-selection first \
    --support-selection-seed 0 \
    --support-transforms identity \
    --support-brightness-range 1.0,1.0 \
    --flow-epochs 3 \
    --coupling-layers 2 \
    --hidden-multiplier 1 \
    --flow-lr 2e-4 \
    --flow-clamp 1.9 \
    --flow-transform-mode flow \
    --tail-weight 0.3 \
    --tail-top-k-ratio 0.05 \
    --lambda-logdet 1e-3 \
    --density-quantile 0.90 \
    --expansion-budget 1.0 \
    --distance-weight 1.0 \
    --density-weight 0.0 \
    --score-mode latent_distance \
    --dvt-denoise-mode none \
    --normality-mode fused \
    --top-percent 0.01 \
    --query-chunk-size 512 \
    --calibration-sample-size 0 2>&1 | tee -a "${log}"

  python3 scripts/run_flow_tte_mvtecad1_guided_refinement.py \
    --data-root "${DATA_ROOT}" \
    --source-root "${raw_root}" \
    --output-root "${guided_root}" \
    --objects "${OBJECTS}" \
    --seed 0 \
    --top-percent 0.01 \
    --cleanup-source-maps \
    --cleanup-output-maps 2>&1 | tee -a "${log}"

  python3 scripts/combine_mvtecad1_hybrid_metrics.py \
    --raw-metrics "${raw_root}/metrics.json" \
    --refined-metrics "${guided_root}/metrics.json" \
    --output "${hybrid_metrics}" 2>&1 | tee -a "${log}"
  echo "[complete] shot=${shot} gpu=${gpu}" | tee -a "${log}"
}

(run_shot 0 1 && run_shot 0 4) & p0=$!
(run_shot 1 2 && run_shot 1 8) & p1=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
echo "[complete] ${RUN_NAME}" | tee "${RUN_ROOT}/remote_run_complete.txt"
