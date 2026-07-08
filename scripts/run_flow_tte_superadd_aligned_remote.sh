#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_superadd_aligned_all8_20260708_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"
BACKBONE_MODEL="${BACKBONE_MODEL:-dinov3_vith16plus}"
FEATURE_LAYERS="${FEATURE_LAYERS:-7,15,23,31}"
BACKBONE_RESOLUTION="${BACKBONE_RESOLUTION:-640}"
TILE_PATCH_SIZE="${TILE_PATCH_SIZE:-640}"
TILE_OVERLAP="${TILE_OVERLAP:-128}"
IMAGE_RESIZE_FACTOR="${IMAGE_RESIZE_FACTOR:-0.625}"
SUPPORT_BRIGHTNESS_RANGE="${SUPPORT_BRIGHTNESS_RANGE:-0.8,1.2}"
SUPPORT_SELECTION="${SUPPORT_SELECTION:-fixed_json=${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json}"
DVT_ALPHA="${DVT_ALPHA:-1.0}"

export FMAD_DINOV3_OFFLINE="${FMAD_DINOV3_OFFLINE:-1}"

mkdir -p "${OUTPUT_ROOT}/chunks" "${OUTPUT_ROOT}/logs"

common_args=(
  --data-root "${DATA_ROOT}"
  --project-root "${PROJECT_ROOT}"
  --fsad-root "${FSAD_ROOT}"
  --shots 16
  --seed 0
  --device cuda
  --flow-epochs 3
  --coupling-layers 2
  --hidden-multiplier 1
  --flow-lr 2e-4
  --flow-clamp 1.9
  --tail-weight 0.3
  --tail-top-k-ratio 0.05
  --lambda-logdet 1e-3
  --density-quantile 0.90
  --expansion-budget 1.0
  --distance-weight 1.0
  --density-weight 0.25
  --score-mode latent_distance
  --top-percent 0.01
  --query-chunk-size 512
  --pro-integration-limit 0.05
  --cleanup-maps
  --backbone-model "${BACKBONE_MODEL}"
  --backbone-resolution "${BACKBONE_RESOLUTION}"
  --feature-layers "${FEATURE_LAYERS}"
  --tile-patch-size "${TILE_PATCH_SIZE}"
  --tile-overlap "${TILE_OVERLAP}"
  --image-resize-factor "${IMAGE_RESIZE_FACTOR}"
  --support-brightness-range "${SUPPORT_BRIGHTNESS_RANGE}"
  --support-selection "${SUPPORT_SELECTION}"
  --support-transforms identity
  --feature-fusion layer_norm_mean
  --context-source none
  --context-mode none
  --flow-condition-mode none
  --flow-transform-mode flow
  --dvt-denoise-mode position_mean
  --dvt-denoise-alpha "${DVT_ALPHA}"
)

run_chunk() {
  local cuda_slot="$1"
  local chunk_name="$2"
  local objects="$3"
  local chunk_root="${OUTPUT_ROOT}/chunks/${chunk_name}"
  local log_path="${OUTPUT_ROOT}/logs/${chunk_name}.log"

  CUDA_VISIBLE_DEVICES="${cuda_slot}" python3 "${FSAD_ROOT}/scripts/run_flow_tte_mvtec_ad2.py" \
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

printf 'run_name=%s\nbackbone=%s\nfeature_layers=%s\nbackbone_resolution=%s\ntile_patch_size=%s\ntile_overlap=%s\nimage_resize_factor=%s\nsupport_brightness_range=%s\nsupport_selection=%s\ndvt_alpha=%s\n' \
  "${RUN_NAME}" "${BACKBONE_MODEL}" "${FEATURE_LAYERS}" "${BACKBONE_RESOLUTION}" \
  "${TILE_PATCH_SIZE}" "${TILE_OVERLAP}" "${IMAGE_RESIZE_FACTOR}" \
  "${SUPPORT_BRIGHTNESS_RANGE}" "${SUPPORT_SELECTION}" "${DVT_ALPHA}" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
