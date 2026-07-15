#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/MVTecAD}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flow_latentbank_mvtecad1_shot4_guided_r8_20260713_v1}"
RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
GPU_COUNT="${GPU_COUNT:-4}"
ALL_OBJECTS="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"

cd "${FSAD_ROOT}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/source_chunks" "${RUN_ROOT}/guided_chunks"
export FMAD_DINOV2_OFFLINE="${FMAD_DINOV2_OFFLINE:-1}"

if [[ "${WAIT_FOR_GRIDSHIFT:-1}" == "1" ]]; then
  while pgrep -f '[r]un_flow_tte_gridshift_2view.py' >/dev/null; do
    echo "[wait] existing grid-shift workers still own dsba3 GPUs"
    sleep 30
  done
fi

run_chunk() {
  local gpu="$1"
  local tag="$2"
  local objects="$3"
  local source="${RUN_ROOT}/source_chunks/${tag}"
  local guided="${RUN_ROOT}/guided_chunks/${tag}"
  local log="${RUN_ROOT}/logs/${tag}.log"
  if [[ -f "${guided}/run_manifest.json" ]]; then
    echo "[skip] ${tag}" | tee -a "${log}"
    return
  fi
  echo "[start] tag=${tag} gpu=${gpu} objects=${objects}" | tee "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/run_flow_tte_mvtec_ad1.py \
    --data-root "${DATA_ROOT}" \
    --output-root "${source}" \
    --project-root /workspace \
    --fsad-root "${FSAD_ROOT}" \
    --objects "${objects}" \
    --shots 4 \
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
    --density-weight 0.25 \
    --score-mode latent_distance \
    --dvt-denoise-mode none \
    --normality-mode fused \
    --top-percent 0.01 \
    --query-chunk-size 512 \
    --calibration-sample-size 0 2>&1 | tee -a "${log}"
  python3 scripts/run_flow_tte_mvtecad1_guided_refinement.py \
    --data-root "${DATA_ROOT}" \
    --source-root "${source}" \
    --output-root "${guided}" \
    --objects "${objects}" \
    --seed 0 \
    --top-percent 0.01 \
    --cleanup-source-maps \
    --cleanup-output-maps 2>&1 | tee -a "${log}"
  echo "[complete] tag=${tag} gpu=${gpu}" | tee -a "${log}"
}

declare -a pids=()
if [[ "${GPU_COUNT}" == "2" ]]; then
  run_chunk 0 gpu0_first_eight "bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut" & pids+=("$!")
  run_chunk 1 gpu1_last_seven "pill,screw,tile,toothbrush,transistor,wood,zipper" & pids+=("$!")
elif [[ "${GPU_COUNT}" == "4" ]]; then
  run_chunk 0 gpu0_bottle_cable_capsule_carpet "bottle,cable,capsule,carpet" & pids+=("$!")
  run_chunk 1 gpu1_grid_hazelnut_leather_metal_nut "grid,hazelnut,leather,metal_nut" & pids+=("$!")
  run_chunk 2 gpu2_pill_screw_tile_toothbrush "pill,screw,tile,toothbrush" & pids+=("$!")
  run_chunk 3 gpu3_transistor_wood_zipper "transistor,wood,zipper" & pids+=("$!")
else
  echo "GPU_COUNT must be 2 or 4, got ${GPU_COUNT}" >&2
  exit 2
fi
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

source_metrics=("${RUN_ROOT}"/source_chunks/*/metrics.json)
guided_metrics=("${RUN_ROOT}"/guided_chunks/*/metrics.json)
python3 scripts/aggregate_mvtecad1_metric_chunks.py \
  --metrics "${source_metrics[@]}" \
  --objects "${ALL_OBJECTS}" \
  --output "${RUN_ROOT}/identity_metrics.json"
python3 scripts/aggregate_mvtecad1_metric_chunks.py \
  --metrics "${guided_metrics[@]}" \
  --objects "${ALL_OBJECTS}" \
  --output "${RUN_ROOT}/guided_r8_metrics.json"
echo "[complete] ${RUN_NAME}" | tee "${RUN_ROOT}/remote_run_complete.txt"
