#!/usr/bin/env bash
set -euo pipefail

RUN_PREFIX="${RUN_PREFIX:-flowtte_patch_structure_all8}"
RUN_SUFFIX="${RUN_SUFFIX:-20260709_v1}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"

run_variant() {
  local variant="$1"
  shift
  local run_name="${RUN_PREFIX}_${variant}_${RUN_SUFFIX}"
  echo "[patch-structure] starting ${run_name}"
  env \
    RUN_NAME="${run_name}" \
    BACKBONE_MODEL="${BACKBONE_MODEL:-dinov3_vith16plus}" \
    FEATURE_LAYERS="${FEATURE_LAYERS:-7,15,23,31}" \
    DVT_DENOISE_MODE="${DVT_DENOISE_MODE:-position_mean}" \
    DVT_ALPHA="${DVT_ALPHA:-1.0}" \
    CLEANUP_MAPS="${CLEANUP_MAPS:-1}" \
    FMAD_DINOV3_OFFLINE="${FMAD_DINOV3_OFFLINE:-1}" \
    "$@" \
    bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"
  echo "[patch-structure] finished ${run_name}"
}

run_variant conditional_cls \
  FLOW_CONDITION_MODE=context \
  FLOW_CONTEXT_SOURCE=cls \
  CONTEXT_MODE=none

run_variant conditional_xy \
  FLOW_CONDITION_MODE=context \
  FLOW_CONTEXT_SOURCE=xy \
  CONTEXT_MODE=none

run_variant conditional_cls_xy \
  FLOW_CONDITION_MODE=context \
  FLOW_CONTEXT_SOURCE=cls_xy \
  CONTEXT_MODE=none

run_variant foreground_flow_mixture \
  NORMALITY_MODE=foreground_flow_mixture \
  SCORE_FIELD_FOREGROUND_QUANTILE="${SCORE_FIELD_FOREGROUND_QUANTILE:-0.50}"

run_variant local_contrast \
  SCORE_FIELD_CALIBRATION_MODE=local_contrast \
  SCORE_FIELD_CALIBRATION_ALPHA="${SCORE_FIELD_CALIBRATION_ALPHA:-1.0}" \
  SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL="${SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL:-7}"
