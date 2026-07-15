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
FLOW_EPOCHS="${FLOW_EPOCHS:-3}"
COUPLING_LAYERS="${COUPLING_LAYERS:-2}"
HIDDEN_MULTIPLIER="${HIDDEN_MULTIPLIER:-1}"
FLOW_LR="${FLOW_LR:-2e-4}"
FLOW_CLAMP="${FLOW_CLAMP:-1.9}"
TAIL_WEIGHT="${TAIL_WEIGHT:-0.3}"
TAIL_TOP_K_RATIO="${TAIL_TOP_K_RATIO:-0.05}"
LAMBDA_LOGDET="${LAMBDA_LOGDET:-1e-3}"
TOP_PERCENT="${TOP_PERCENT:-0.01}"
QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-512}"
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
CALIBRATION_SAMPLE_SIZE="${CALIBRATION_SAMPLE_SIZE:-0}"
FLOW_CONDITION_MODE="${FLOW_CONDITION_MODE:-none}"
TRANSFORMER_CONTEXT_MODE="${TRANSFORMER_CONTEXT_MODE:-none}"
BACKBONE_RESOLUTION="${BACKBONE_RESOLUTION:-0}"
TILE_PATCH_SIZE="${TILE_PATCH_SIZE:-0}"
TILE_OVERLAP="${TILE_OVERLAP:-0}"
IMAGE_RESIZE_FACTOR="${IMAGE_RESIZE_FACTOR:-1.0}"
SUPPORT_BRIGHTNESS_RANGE="${SUPPORT_BRIGHTNESS_RANGE:-1.0,1.0}"
FEATURE_FUSION="${FEATURE_FUSION:-layer_norm_mean}"
SUPPORT_TRANSFORMS="${SUPPORT_TRANSFORMS:-identity}"
SMOKE_OBJECT="${SMOKE_OBJECT:-}"
SMOKE_CUDA_SLOT="${SMOKE_CUDA_SLOT:-0}"
EXPECTED_METHOD_BUNDLE_SHA256="${METHOD_BUNDLE_SHA256:-}"
mapfile -t METHOD_BUNDLE_FILES < <(
  {
    find "${FSAD_ROOT}/src/flow_tte" -maxdepth 1 -type f -name '*.py' -print
    find "${FSAD_ROOT}/scripts" -maxdepth 1 -type f -name 'flow_tte_*.py' -print
    find "${PROJECT_ROOT}/fmad/datasets" -maxdepth 1 -type f -name '*.py' -print
    find "${PROJECT_ROOT}/fmad/evaluation" -maxdepth 1 -type f -name '*.py' -print
    printf '%s\n' \
      "${FSAD_ROOT}/scripts/dinov3_backbone.py" \
      "${FSAD_ROOT}/scripts/run_flow_tte_mvtec_ad2.py" \
      "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh" \
      "${FSAD_ROOT}/scripts/run_flow_tte_hplus_rotation8_remote.sh" \
      "${FSAD_ROOT}/scripts/run_flow_tte_hplus_rotation8_ablation_remote.sh" \
      "${PROJECT_ROOT}/fmad/registry.py" \
      "${PROJECT_ROOT}/src/post_eval.py" \
      "${PROJECT_ROOT}/src/utils.py"
  } | sort -u
)
METHOD_BUNDLE_SHA256="$(
  sha256sum "${METHOD_BUNDLE_FILES[@]}" \
    | awk '{print $1}' \
    | sha256sum \
    | awk '{print $1}'
)"
if [[ -n "${EXPECTED_METHOD_BUNDLE_SHA256}" && "${METHOD_BUNDLE_SHA256}" != "${EXPECTED_METHOD_BUNDLE_SHA256}" ]]; then
  echo "method bundle changed after controller initialization" >&2
  exit 1
fi

export FMAD_DINOV3_OFFLINE="${FMAD_DINOV3_OFFLINE:-1}"

mkdir -p "${OUTPUT_ROOT}/chunks" "${OUTPUT_ROOT}/logs"

common_args=(
  --data-root "${DATA_ROOT}"
  --project-root "${PROJECT_ROOT}"
  --fsad-root "${FSAD_ROOT}"
  --shots 16
  --seed 0
  --device cuda
  --flow-epochs "${FLOW_EPOCHS}"
  --coupling-layers "${COUPLING_LAYERS}"
  --hidden-multiplier "${HIDDEN_MULTIPLIER}"
  --flow-lr "${FLOW_LR}"
  --flow-clamp "${FLOW_CLAMP}"
  --tail-weight "${TAIL_WEIGHT}"
  --tail-top-k-ratio "${TAIL_TOP_K_RATIO}"
  --lambda-logdet "${LAMBDA_LOGDET}"
  --density-quantile 0.90
  --expansion-budget 1.0
  --distance-weight 1.0
  --density-weight "${DENSITY_WEIGHT}"
  --score-mode "${SCORE_MODE}"
  --residual-weight "${RESIDUAL_WEIGHT}"
  --top-percent "${TOP_PERCENT}"
  --query-chunk-size "${QUERY_CHUNK_SIZE}"
  --pro-integration-limit 0.05
  --rgb-guide none
  --backbone-model "${BACKBONE_MODEL}"
  --backbone-resolution "${BACKBONE_RESOLUTION}"
  --feature-layers "${FEATURE_LAYERS}"
  --tile-patch-size "${TILE_PATCH_SIZE}"
  --tile-overlap "${TILE_OVERLAP}"
  --image-resize-factor "${IMAGE_RESIZE_FACTOR}"
  --support-brightness-range "${SUPPORT_BRIGHTNESS_RANGE}"
  --support-selection "${SUPPORT_SELECTION}"
  --support-transforms "${SUPPORT_TRANSFORMS}"
  --feature-fusion "${FEATURE_FUSION}"
  --normality-mode "${NORMALITY_MODE}"
  --context-source "${CONTEXT_SOURCE}"
  --flow-context-source "${FLOW_CONTEXT_SOURCE}"
  --memory-context-source "${MEMORY_CONTEXT_SOURCE}"
  --context-mode "${CONTEXT_MODE}"
  --context-weight "${CONTEXT_WEIGHT}"
  --context-top-m "${CONTEXT_TOP_M}"
  --calibration-sample-size "${CALIBRATION_SAMPLE_SIZE}"
  --flow-condition-mode "${FLOW_CONDITION_MODE}"
  --transformer-context-mode "${TRANSFORMER_CONTEXT_MODE}"
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
  local execution_mode="$1"
  local cuda_slot="$2"
  local chunk_name="$3"
  local objects="$4"
  local chunk_root="${OUTPUT_ROOT}/chunks/${chunk_name}"
  local log_path="${OUTPUT_ROOT}/logs/${chunk_name}.log"

  if [[ "${execution_mode}" == "replace" ]]; then
    exec env CUDA_VISIBLE_DEVICES="${cuda_slot}" \
      python3 "${FSAD_ROOT}/scripts/run_flow_tte_mvtec_ad2.py" \
      "${common_args[@]}" \
      --output-root "${chunk_root}" \
      --objects "${objects}" \
      >"${log_path}" 2>&1
  fi
  if [[ "${execution_mode}" != "foreground" ]]; then
    echo "unknown chunk execution mode: ${execution_mode}" >&2
    return 2
  fi
  env CUDA_VISIBLE_DEVICES="${cuda_slot}" \
    python3 "${FSAD_ROOT}/scripts/run_flow_tte_mvtec_ad2.py" \
    "${common_args[@]}" \
    --output-root "${chunk_root}" \
    --objects "${objects}" \
    >"${log_path}" 2>&1
}

if [[ -n "${SMOKE_OBJECT}" ]]; then
  run_chunk foreground "${SMOKE_CUDA_SLOT}" "smoke_${SMOKE_OBJECT}" "${SMOKE_OBJECT}"
else
  child_pids=()
  cleanup_children() {
    local pid
    for pid in "${child_pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" 2>/dev/null || true
      fi
    done
    wait "${child_pids[@]}" 2>/dev/null || true
  }
  trap cleanup_children EXIT
  trap 'cleanup_children; exit 1' INT TERM

  run_chunk replace 0 gpu0_can_fabric_fruit_jelly can,fabric,fruit_jelly &
  pid0=$!
  child_pids+=("${pid0}")
  run_chunk replace 1 gpu1_rice_vial_wallplugs rice,vial,wallplugs &
  pid1=$!
  child_pids+=("${pid1}")
  run_chunk replace 2 gpu2_walnuts_sheet_metal walnuts,sheet_metal &
  pid2=$!
  child_pids+=("${pid2}")

  wait_status=0
  wait "${pid0}" || wait_status=$?
  wait "${pid1}" || wait_status=$?
  wait "${pid2}" || wait_status=$?
  trap - EXIT INT TERM
  if [[ "${wait_status}" -ne 0 ]]; then
    exit "${wait_status}"
  fi
fi

printf 'run_name=%s\nmethod_bundle_sha256=%s\nbackbone=%s\nbackbone_resolution=%s\nfeature_layers=%s\ntile_patch_size=%s\ntile_overlap=%s\nimage_resize_factor=%s\nsupport_brightness_range=%s\nfeature_fusion=%s\nsupport_selection=%s\nsupport_transforms=%s\ndvt_denoise_mode=%s\ndvt_alpha=%s\nnormality_mode=%s\ncontext_source=%s\nflow_context_source=%s\nmemory_context_source=%s\ncontext_mode=%s\ncontext_weight=%s\ncontext_top_m=%s\ncalibration_sample_size=%s\nflow_condition_mode=%s\ntransformer_context_mode=%s\nflow_transform_mode=%s\nflow_epochs=%s\ncoupling_layers=%s\nhidden_multiplier=%s\nflow_lr=%s\nflow_clamp=%s\ntail_weight=%s\ntail_top_k_ratio=%s\nlambda_logdet=%s\ntop_percent=%s\nquery_chunk_size=%s\ndensity_weight=%s\nscore_mode=%s\nresidual_weight=%s\nscore_field_calibration_mode=%s\nscore_field_calibration_alpha=%s\nscore_field_position_std_floor=%s\nscore_field_foreground_mode=%s\nscore_field_foreground_quantile=%s\nscore_field_background_multiplier=%s\nscore_field_foreground_smooth_kernel=%s\nscore_field_support_score_quantile=%s\n' \
  "${RUN_NAME}" "${METHOD_BUNDLE_SHA256}" "${BACKBONE_MODEL}" "${BACKBONE_RESOLUTION}" "${FEATURE_LAYERS}" \
  "${TILE_PATCH_SIZE}" "${TILE_OVERLAP}" "${IMAGE_RESIZE_FACTOR}" "${SUPPORT_BRIGHTNESS_RANGE}" \
  "${FEATURE_FUSION}" "${SUPPORT_SELECTION}" "${SUPPORT_TRANSFORMS}" "${DVT_DENOISE_MODE}" "${DVT_ALPHA}" \
  "${NORMALITY_MODE}" "${CONTEXT_SOURCE}" "${FLOW_CONTEXT_SOURCE}" "${MEMORY_CONTEXT_SOURCE}" \
  "${CONTEXT_MODE}" "${CONTEXT_WEIGHT}" "${CONTEXT_TOP_M}" "${CALIBRATION_SAMPLE_SIZE}" "${FLOW_CONDITION_MODE}" \
  "${TRANSFORMER_CONTEXT_MODE}" "${FLOW_TRANSFORM_MODE}" "${FLOW_EPOCHS}" "${COUPLING_LAYERS}" \
  "${HIDDEN_MULTIPLIER}" "${FLOW_LR}" "${FLOW_CLAMP}" "${TAIL_WEIGHT}" "${TAIL_TOP_K_RATIO}" \
  "${LAMBDA_LOGDET}" "${TOP_PERCENT}" "${QUERY_CHUNK_SIZE}" \
  "${DENSITY_WEIGHT}" "${SCORE_MODE}" "${RESIDUAL_WEIGHT}" \
  "${SCORE_FIELD_CALIBRATION_MODE}" "${SCORE_FIELD_CALIBRATION_ALPHA}" \
  "${SCORE_FIELD_POSITION_STD_FLOOR}" "${SCORE_FIELD_FOREGROUND_MODE}" \
  "${SCORE_FIELD_FOREGROUND_QUANTILE}" "${SCORE_FIELD_BACKGROUND_MULTIPLIER}" \
  "${SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL}" "${SCORE_FIELD_SUPPORT_SCORE_QUANTILE}" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
