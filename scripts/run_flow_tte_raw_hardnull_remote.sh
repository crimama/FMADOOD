#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_raw_hardnull_all8_20260708_v1}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"
SUPPORT_SELECTION="${SUPPORT_SELECTION:-fixed_json=${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json}"

mkdir -p "${OUTPUT_ROOT}/variants"
: >"${OUTPUT_ROOT}/variant_roots.tsv"
printf 'variant\troot\n' >"${OUTPUT_ROOT}/variant_roots.tsv"

run_variant() {
  local variant="$1"
  local variant_root="${OUTPUT_ROOT}/variants/${variant}"
  shift
  printf '%s\t%s\n' "${variant}" "${variant_root}" >>"${OUTPUT_ROOT}/variant_roots.tsv"
  RUN_NAME="${RUN_NAME}_${variant}" \
  OUTPUT_ROOT="${variant_root}" \
  BACKBONE_MODEL=dinov3_vith16plus \
  FEATURE_LAYERS=7,15,23,31 \
  SUPPORT_SELECTION="${SUPPORT_SELECTION}" \
  CLEANUP_MAPS=1 \
  FMAD_DINOV3_OFFLINE=1 \
  "$@" \
  bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"
}

run_variant raw_layerwise_tiled_no_dvt \
  env NORMALITY_MODE=raw_layer_wise DVT_DENOISE_MODE=none DVT_ALPHA=0.0 \
    BACKBONE_RESOLUTION=640 TILE_PATCH_SIZE=640 TILE_OVERLAP=128 IMAGE_RESIZE_FACTOR=1.0 \
    DENSITY_WEIGHT=0.0

run_variant raw_layerwise_tiled_dvt \
  env NORMALITY_MODE=raw_layer_wise DVT_DENOISE_MODE=position_mean DVT_ALPHA=1.0 \
    BACKBONE_RESOLUTION=640 TILE_PATCH_SIZE=640 TILE_OVERLAP=128 IMAGE_RESIZE_FACTOR=1.0 \
    DENSITY_WEIGHT=0.0

run_variant raw_fused_dvt \
  env NORMALITY_MODE=raw_nn DVT_DENOISE_MODE=position_mean DVT_ALPHA=1.0 \
    BACKBONE_RESOLUTION=0 TILE_PATCH_SIZE=0 TILE_OVERLAP=0 IMAGE_RESIZE_FACTOR=1.0 \
    DENSITY_WEIGHT=0.0

run_variant raw_nn_nf_residual_dvt \
  env NORMALITY_MODE=raw_nn_nf_residual DVT_DENOISE_MODE=position_mean DVT_ALPHA=1.0 \
    BACKBONE_RESOLUTION=0 TILE_PATCH_SIZE=0 TILE_OVERLAP=0 IMAGE_RESIZE_FACTOR=1.0 \
    DENSITY_WEIGHT=0.25 RESIDUAL_WEIGHT=0.25

run_variant foreground_raw_fused_dvt \
  env NORMALITY_MODE=foreground_raw_nn DVT_DENOISE_MODE=position_mean DVT_ALPHA=1.0 \
    BACKBONE_RESOLUTION=0 TILE_PATCH_SIZE=0 TILE_OVERLAP=0 IMAGE_RESIZE_FACTOR=1.0 \
    DENSITY_WEIGHT=0.0 SCORE_FIELD_FOREGROUND_QUANTILE=0.5 \
    SCORE_FIELD_BACKGROUND_MULTIPLIER=0.5

find "${OUTPUT_ROOT}" -type d -name anomaly_maps -prune -exec rm -rf {} +
printf 'run_name=%s\ncleanup_anomaly_maps=true\n' "${RUN_NAME}" \
  >"${OUTPUT_ROOT}/raw_hardnull_remote_complete.txt"
