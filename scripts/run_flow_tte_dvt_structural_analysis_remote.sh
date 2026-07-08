#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_dvt_structural_analysis_20260708_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"
SUPPORT_JSON="${SUPPORT_JSON:-${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json}"
DVT_ALPHA="${DVT_ALPHA:-1.0}"
TEST_IMAGES_PER_SPLIT="${TEST_IMAGES_PER_SPLIT:-12}"
SUPPORT_SAMPLE_PATCHES="${SUPPORT_SAMPLE_PATCHES:-4096}"

export FMAD_DINOV3_OFFLINE="${FMAD_DINOV3_OFFLINE:-1}"

mkdir -p "${OUTPUT_ROOT}/chunks" "${OUTPUT_ROOT}/logs"

common_args=(
  --data-root "${DATA_ROOT}"
  --project-root "${PROJECT_ROOT}"
  --fsad-root "${FSAD_ROOT}"
  --support-json "${SUPPORT_JSON}"
  --device cuda
  --seed 0
  --alpha "${DVT_ALPHA}"
  --top-percent 0.01
  --test-images-per-split "${TEST_IMAGES_PER_SPLIT}"
  --support-sample-patches "${SUPPORT_SAMPLE_PATCHES}"
  --query-chunk-size 256
)

run_chunk() {
  local cuda_slot="$1"
  local chunk_name="$2"
  local objects="$3"
  local chunk_root="${OUTPUT_ROOT}/chunks/${chunk_name}"
  local log_path="${OUTPUT_ROOT}/logs/${chunk_name}.log"

  CUDA_VISIBLE_DEVICES="${cuda_slot}" python3 "${FSAD_ROOT}/scripts/run_flow_tte_dvt_structural_analysis.py" \
    "${common_args[@]}" \
    --output-root "${chunk_root}" \
    --objects "${objects}" \
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

printf 'run_name=%s\nsupport_json=%s\ndvt_alpha=%s\ntest_images_per_split=%s\n' \
  "${RUN_NAME}" "${SUPPORT_JSON}" "${DVT_ALPHA}" "${TEST_IMAGES_PER_SPLIT}" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
