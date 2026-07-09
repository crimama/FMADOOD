#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_dvt_denoising_all8_a10_20260707_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"
BACKBONE_MODEL="${BACKBONE_MODEL:-dinov3_vitl16}"
FEATURE_LAYERS="${FEATURE_LAYERS:-5,11,17,23}"
SUPPORT_SELECTION="${SUPPORT_SELECTION:-fixed_json=${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json}"
DVT_ALPHA="${DVT_ALPHA:-1.0}"
DVT_DENOISE_MODE="${DVT_DENOISE_MODE:-position_mean}"
CLEANUP_MAPS="${CLEANUP_MAPS:-1}"
FLOW_TRANSFORM_MODE="${FLOW_TRANSFORM_MODE:-flow}"
DENSITY_WEIGHT="${DENSITY_WEIGHT:-0.25}"
SCORE_MODE="${SCORE_MODE:-latent_distance}"
RESIDUAL_WEIGHT="${RESIDUAL_WEIGHT:-0.25}"
SCORE_FIELD_CALIBRATION_MODE="${SCORE_FIELD_CALIBRATION_MODE:-none}"
SCORE_FIELD_CALIBRATION_ALPHA="${SCORE_FIELD_CALIBRATION_ALPHA:-1.0}"
SCORE_FIELD_POSITION_STD_FLOOR="${SCORE_FIELD_POSITION_STD_FLOOR:-0.25}"
SCORE_FIELD_FOREGROUND_MODE="${SCORE_FIELD_FOREGROUND_MODE:-none}"
SCORE_FIELD_FOREGROUND_QUANTILE="${SCORE_FIELD_FOREGROUND_QUANTILE:-0.20}"
SCORE_FIELD_BACKGROUND_MULTIPLIER="${SCORE_FIELD_BACKGROUND_MULTIPLIER:-0.50}"
SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL="${SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL:-5}"
SCORE_FIELD_SUPPORT_SCORE_QUANTILE="${SCORE_FIELD_SUPPORT_SCORE_QUANTILE:-0.90}"
NORMALITY_MODE="${NORMALITY_MODE:-fused}"
CONTEXT_SOURCE="${CONTEXT_SOURCE:-none}"
FLOW_CONTEXT_SOURCE="${FLOW_CONTEXT_SOURCE:-auto}"
MEMORY_CONTEXT_SOURCE="${MEMORY_CONTEXT_SOURCE:-auto}"
CONTEXT_MODE="${CONTEXT_MODE:-none}"
CONTEXT_WEIGHT="${CONTEXT_WEIGHT:-0.0}"
CONTEXT_TOP_M="${CONTEXT_TOP_M:-1}"
FLOW_CONDITION_MODE="${FLOW_CONDITION_MODE:-none}"
BACKBONE_RESOLUTION="${BACKBONE_RESOLUTION:-0}"
TILE_PATCH_SIZE="${TILE_PATCH_SIZE:-0}"
TILE_OVERLAP="${TILE_OVERLAP:-0}"
IMAGE_RESIZE_FACTOR="${IMAGE_RESIZE_FACTOR:-1.0}"
SUPPORT_BRIGHTNESS_RANGE="${SUPPORT_BRIGHTNESS_RANGE:-1.0,1.0}"
FEATURE_FUSION="${FEATURE_FUSION:-layer_norm_mean}"

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
  --density-weight "${DENSITY_WEIGHT}"
  --score-mode "${SCORE_MODE}"
  --residual-weight "${RESIDUAL_WEIGHT}"
  --top-percent 0.01
  --query-chunk-size 512
  --pro-integration-limit 0.05
  --backbone-model "${BACKBONE_MODEL}"
  --backbone-resolution "${BACKBONE_RESOLUTION}"
  --feature-layers "${FEATURE_LAYERS}"
  --tile-patch-size "${TILE_PATCH_SIZE}"
  --tile-overlap "${TILE_OVERLAP}"
  --image-resize-factor "${IMAGE_RESIZE_FACTOR}"
  --support-brightness-range "${SUPPORT_BRIGHTNESS_RANGE}"
  --support-selection "${SUPPORT_SELECTION}"
  --support-transforms identity
  --feature-fusion "${FEATURE_FUSION}"
  --normality-mode "${NORMALITY_MODE}"
  --context-source "${CONTEXT_SOURCE}"
  --flow-context-source "${FLOW_CONTEXT_SOURCE}"
  --memory-context-source "${MEMORY_CONTEXT_SOURCE}"
  --context-mode "${CONTEXT_MODE}"
  --context-weight "${CONTEXT_WEIGHT}"
  --context-top-m "${CONTEXT_TOP_M}"
  --flow-condition-mode "${FLOW_CONDITION_MODE}"
  --flow-transform-mode "${FLOW_TRANSFORM_MODE}"
  --dvt-denoise-mode "${DVT_DENOISE_MODE}"
  --dvt-denoise-alpha "${DVT_ALPHA}"
  --score-field-calibration-mode "${SCORE_FIELD_CALIBRATION_MODE}"
  --score-field-calibration-alpha "${SCORE_FIELD_CALIBRATION_ALPHA}"
  --score-field-position-std-floor "${SCORE_FIELD_POSITION_STD_FLOOR}"
  --score-field-foreground-mode "${SCORE_FIELD_FOREGROUND_MODE}"
  --score-field-foreground-quantile "${SCORE_FIELD_FOREGROUND_QUANTILE}"
  --score-field-background-multiplier "${SCORE_FIELD_BACKGROUND_MULTIPLIER}"
  --score-field-foreground-smooth-kernel "${SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL}"
  --score-field-support-score-quantile "${SCORE_FIELD_SUPPORT_SCORE_QUANTILE}"
)

if [[ "${CLEANUP_MAPS}" == "1" ]]; then
  common_args+=(--cleanup-maps)
fi

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

printf 'run_name=%s\nbackbone=%s\nbackbone_resolution=%s\nfeature_layers=%s\ntile_patch_size=%s\ntile_overlap=%s\nimage_resize_factor=%s\nsupport_brightness_range=%s\nfeature_fusion=%s\nsupport_selection=%s\ndvt_denoise_mode=%s\ndvt_alpha=%s\nnormality_mode=%s\ncontext_source=%s\nflow_context_source=%s\nmemory_context_source=%s\ncontext_mode=%s\ncontext_weight=%s\ncontext_top_m=%s\nflow_condition_mode=%s\nflow_transform_mode=%s\ndensity_weight=%s\nscore_mode=%s\nresidual_weight=%s\nscore_field_calibration_mode=%s\nscore_field_calibration_alpha=%s\nscore_field_position_std_floor=%s\nscore_field_foreground_mode=%s\nscore_field_foreground_quantile=%s\nscore_field_background_multiplier=%s\nscore_field_foreground_smooth_kernel=%s\nscore_field_support_score_quantile=%s\n' \
  "${RUN_NAME}" "${BACKBONE_MODEL}" "${BACKBONE_RESOLUTION}" "${FEATURE_LAYERS}" \
  "${TILE_PATCH_SIZE}" "${TILE_OVERLAP}" "${IMAGE_RESIZE_FACTOR}" "${SUPPORT_BRIGHTNESS_RANGE}" \
  "${FEATURE_FUSION}" "${SUPPORT_SELECTION}" "${DVT_DENOISE_MODE}" "${DVT_ALPHA}" \
  "${NORMALITY_MODE}" "${CONTEXT_SOURCE}" "${FLOW_CONTEXT_SOURCE}" "${MEMORY_CONTEXT_SOURCE}" \
  "${CONTEXT_MODE}" "${CONTEXT_WEIGHT}" "${CONTEXT_TOP_M}" "${FLOW_CONDITION_MODE}" \
  "${FLOW_TRANSFORM_MODE}" "${DENSITY_WEIGHT}" "${SCORE_MODE}" "${RESIDUAL_WEIGHT}" \
  "${SCORE_FIELD_CALIBRATION_MODE}" "${SCORE_FIELD_CALIBRATION_ALPHA}" \
  "${SCORE_FIELD_POSITION_STD_FLOOR}" "${SCORE_FIELD_FOREGROUND_MODE}" \
  "${SCORE_FIELD_FOREGROUND_QUANTILE}" "${SCORE_FIELD_BACKGROUND_MULTIPLIER}" \
  "${SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL}" "${SCORE_FIELD_SUPPORT_SCORE_QUANTILE}" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
