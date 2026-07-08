#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_hplus_postprocess_all8_20260708_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"

RUN_NAME="${RUN_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
BACKBONE_MODEL=dinov3_vith16plus \
FEATURE_LAYERS=7,15,23,31 \
DVT_ALPHA=1.0 \
CLEANUP_MAPS=0 \
FMAD_DINOV3_OFFLINE=1 \
bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"

python3 "${FSAD_ROOT}/scripts/flow_tte_postprocess_eval.py" \
  --data-root "${DATA_ROOT}" \
  --objects can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal \
  --run-root "${OUTPUT_ROOT}/chunks/gpu0_can_fabric_fruit_jelly" \
  --run-root "${OUTPUT_ROOT}/chunks/gpu1_rice_vial_wallplugs" \
  --run-root "${OUTPUT_ROOT}/chunks/gpu2_walnuts_sheet_metal" \
  --output-json "${OUTPUT_ROOT}/postprocess_eval.json" \
  --output-tsv "${OUTPUT_ROOT}/postprocess_eval.tsv"

find "${OUTPUT_ROOT}" -type d -name anomaly_maps -prune -exec rm -rf {} +
printf 'cleanup_anomaly_maps=true\n' >"${OUTPUT_ROOT}/cleanup_evidence.txt"
printf 'postprocess_eval=%s\ncleanup_maps=true\n' "${OUTPUT_ROOT}/postprocess_eval.json" \
  >>"${OUTPUT_ROOT}/remote_run_complete.txt"
