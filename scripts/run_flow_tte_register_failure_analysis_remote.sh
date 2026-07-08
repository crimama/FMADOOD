#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
SUPPORT_JSON="${SUPPORT_JSON:?SUPPORT_JSON is required}"

DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"
PATCH_SAMPLES_PER_IMAGE="${PATCH_SAMPLES_PER_IMAGE:-128}"
CONTEXT_TOP_M="${CONTEXT_TOP_M:-4}"

export FMAD_DINOV3_OFFLINE="${FMAD_DINOV3_OFFLINE:-1}"

mkdir -p "${OUTPUT_ROOT}/chunks" "${OUTPUT_ROOT}/logs"

run_chunk() {
  local cuda_slot="$1"
  local chunk_name="$2"
  local objects="$3"
  local chunk_root="${OUTPUT_ROOT}/chunks/${chunk_name}"
  local log_path="${OUTPUT_ROOT}/logs/${chunk_name}.log"

  CUDA_VISIBLE_DEVICES="${cuda_slot}" python3 "${FSAD_ROOT}/scripts/run_flow_tte_register_failure_analysis.py" \
    --data-root "${DATA_ROOT}" \
    --output-root "${chunk_root}" \
    --project-root "${PROJECT_ROOT}" \
    --fsad-root "${FSAD_ROOT}" \
    --support-json "${SUPPORT_JSON}" \
    --objects "${objects}" \
    --device cuda \
    --seed 0 \
    --patch-samples-per-image "${PATCH_SAMPLES_PER_IMAGE}" \
    --context-top-m "${CONTEXT_TOP_M}" \
    >"${log_path}" 2>&1
}

run_chunk 0 gpu0_can_fabric_fruit_jelly can,fabric,fruit_jelly &
pid0=$!
run_chunk 1 gpu1_rice_vial_wallplugs rice,vial,wallplugs &
pid1=$!
run_chunk 2 gpu2_walnuts_sheet_metal walnuts,sheet_metal &
pid2=$!

wait "${pid0}"
wait "${pid1}"
wait "${pid2}"

printf 'run_name=%s\nsupport_json=%s\npatch_samples_per_image=%s\ncontext_top_m=%s\n' \
  "${RUN_NAME}" "${SUPPORT_JSON}" "${PATCH_SAMPLES_PER_IMAGE}" "${CONTEXT_TOP_M}" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
