#!/usr/bin/env bash
set -euo pipefail

RUN_GROUP_NAME="${RUN_GROUP_NAME:-flowtte_scorefield_structural_all8_20260708_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
GROUP_ROOT="${RESULTS_ROOT}/${RUN_GROUP_NAME}"
OBJECTS="can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-0}"

mkdir -p "${GROUP_ROOT}"

run_variant() {
  local variant_name="$1"
  local calibration_mode="$2"
  local foreground_mode="$3"
  local run_name="${RUN_GROUP_NAME}_${variant_name}"
  local output_root="${GROUP_ROOT}/${variant_name}"

  RUN_NAME="${run_name}" \
  OUTPUT_ROOT="${output_root}" \
  BACKBONE_MODEL=dinov3_vith16plus \
  FEATURE_LAYERS=7,15,23,31 \
  DVT_ALPHA=1.0 \
  CLEANUP_MAPS=0 \
  SCORE_FIELD_CALIBRATION_MODE="${calibration_mode}" \
  SCORE_FIELD_CALIBRATION_ALPHA=1.0 \
  SCORE_FIELD_POSITION_STD_FLOOR=0.25 \
  SCORE_FIELD_FOREGROUND_MODE="${foreground_mode}" \
  SCORE_FIELD_FOREGROUND_QUANTILE=0.20 \
  SCORE_FIELD_BACKGROUND_MULTIPLIER=0.50 \
  SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL=5 \
  FMAD_DINOV3_OFFLINE=1 \
  bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"

  if [[ "${RUN_POSTPROCESS}" == "1" ]]; then
    python3 "${FSAD_ROOT}/scripts/flow_tte_postprocess_eval.py" \
      --data-root "${DATA_ROOT}" \
      --objects "${OBJECTS}" \
      --run-root "${output_root}/chunks/gpu0_can_fabric_fruit_jelly" \
      --run-root "${output_root}/chunks/gpu1_rice_vial_wallplugs" \
      --run-root "${output_root}/chunks/gpu2_walnuts_sheet_metal" \
      --threshold-count 1 \
      --output-json "${output_root}/postprocess_eval.json" \
      --output-tsv "${output_root}/postprocess_eval.tsv"
  else
    printf 'postprocess_eval=skipped\nreason=score_field_structure_experiment\n' \
      >"${output_root}/postprocess_skipped.txt"
  fi

  python3 "${FSAD_ROOT}/scripts/flow_tte_score_field_analysis.py" \
    --data-root "${DATA_ROOT}" \
    --objects "${OBJECTS}" \
    --run-root "${output_root}/chunks/gpu0_can_fabric_fruit_jelly" \
    --run-root "${output_root}/chunks/gpu1_rice_vial_wallplugs" \
    --run-root "${output_root}/chunks/gpu2_walnuts_sheet_metal" \
    --output-json "${output_root}/fragmentation_analysis.json" \
    --output-tsv "${output_root}/fragmentation_analysis.tsv"

  find "${output_root}" -type d -name anomaly_maps -prune -exec rm -rf {} +
  printf 'variant=%s\ncleanup_anomaly_maps=true\n' "${variant_name}" \
    >"${output_root}/cleanup_evidence.txt"
}

run_variant baseline none none
run_variant support_position_center support_position_center none
run_variant support_position_zscore support_position_zscore none
run_variant foreground_energy none support_feature_energy
run_variant center_plus_foreground support_position_center support_feature_energy

printf 'run_group=%s\nvariants=%s\ncleanup_maps=true\nrun_postprocess=%s\n' \
  "${RUN_GROUP_NAME}" \
  "baseline,support_position_center,support_position_zscore,foreground_energy,center_plus_foreground" \
  "${RUN_POSTPROCESS}" \
  >"${GROUP_ROOT}/remote_run_complete.txt"
