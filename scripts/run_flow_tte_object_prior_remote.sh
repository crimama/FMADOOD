#!/usr/bin/env bash
set -euo pipefail

RUN_GROUP_NAME="${RUN_GROUP_NAME:-flowtte_object_prior_all8_20260709_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
GROUP_ROOT="${RESULTS_ROOT}/${RUN_GROUP_NAME}"

mkdir -p "${GROUP_ROOT}"
: >"${GROUP_ROOT}/variant_roots.tsv"
printf 'variant\troot\n' >"${GROUP_ROOT}/variant_roots.tsv"

run_variant() {
  local variant_name="$1"
  shift
  local output_root="${GROUP_ROOT}/${variant_name}"
  printf '%s\t%s\n' "${variant_name}" "${output_root}" >>"${GROUP_ROOT}/variant_roots.tsv"
  RUN_NAME="${RUN_GROUP_NAME}_${variant_name}" \
  OUTPUT_ROOT="${output_root}" \
  BACKBONE_MODEL=dinov3_vith16plus \
  FEATURE_LAYERS=7,15,23,31 \
  DVT_ALPHA=1.0 \
  NORMALITY_MODE=fused \
  CLEANUP_MAPS=1 \
  FMAD_DINOV3_OFFLINE=1 \
  "$@" \
  bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"
}

run_variant rgb_object_prior \
  env SCORE_FIELD_CALIBRATION_MODE=none \
    SCORE_FIELD_FOREGROUND_MODE=support_rgb_contrast \
    SCORE_FIELD_FOREGROUND_QUANTILE=0.70 \
    SCORE_FIELD_BACKGROUND_MULTIPLIER=0.35 \
    SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL=7

run_variant rgb_feature_product_prior \
  env SCORE_FIELD_CALIBRATION_MODE=none \
    SCORE_FIELD_FOREGROUND_MODE=support_rgb_feature_product \
    SCORE_FIELD_FOREGROUND_QUANTILE=0.70 \
    SCORE_FIELD_BACKGROUND_MULTIPLIER=0.35 \
    SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL=7

run_variant support_score_reliability \
  env SCORE_FIELD_CALIBRATION_MODE=support_score_reliability \
    SCORE_FIELD_CALIBRATION_ALPHA=0.35 \
    SCORE_FIELD_SUPPORT_SCORE_QUANTILE=0.90 \
    SCORE_FIELD_FOREGROUND_MODE=none

run_variant rgb_prior_plus_reliability \
  env SCORE_FIELD_CALIBRATION_MODE=support_score_reliability \
    SCORE_FIELD_CALIBRATION_ALPHA=0.35 \
    SCORE_FIELD_SUPPORT_SCORE_QUANTILE=0.90 \
    SCORE_FIELD_FOREGROUND_MODE=support_rgb_contrast \
    SCORE_FIELD_FOREGROUND_QUANTILE=0.70 \
    SCORE_FIELD_BACKGROUND_MULTIPLIER=0.35 \
    SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL=7

find "${GROUP_ROOT}" -type d -name anomaly_maps -prune -exec rm -rf {} +
printf 'run_group=%s\nvariants=%s\ncleanup_anomaly_maps=true\n' \
  "${RUN_GROUP_NAME}" \
  "rgb_object_prior,rgb_feature_product_prior,support_score_reliability,rgb_prior_plus_reliability" \
  >"${GROUP_ROOT}/remote_run_complete.txt"
