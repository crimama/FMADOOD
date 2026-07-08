#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_layerwise_ctx_cls_topm4_all8_20260708_v1}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"

RUN_NAME="${RUN_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
BACKBONE_MODEL=dinov3_vith16plus \
FEATURE_LAYERS=7,15,23,31 \
DVT_ALPHA=1.0 \
NORMALITY_MODE=layer_wise \
CONTEXT_SOURCE="${CONTEXT_SOURCE:-cls}" \
MEMORY_CONTEXT_SOURCE="${MEMORY_CONTEXT_SOURCE:-auto}" \
FLOW_CONTEXT_SOURCE="${FLOW_CONTEXT_SOURCE:-auto}" \
CONTEXT_MODE="${CONTEXT_MODE:-top_m}" \
CONTEXT_TOP_M="${CONTEXT_TOP_M:-4}" \
CONTEXT_WEIGHT="${CONTEXT_WEIGHT:-0.0}" \
FLOW_CONDITION_MODE="${FLOW_CONDITION_MODE:-none}" \
SCORE_FIELD_CALIBRATION_MODE="${SCORE_FIELD_CALIBRATION_MODE:-none}" \
SCORE_FIELD_FOREGROUND_MODE="${SCORE_FIELD_FOREGROUND_MODE:-none}" \
CLEANUP_MAPS=0 \
FMAD_DINOV3_OFFLINE=1 \
bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"

python3 "${FSAD_ROOT}/scripts/flow_tte_score_field_analysis.py" \
  --data-root "${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}" \
  --objects can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal \
  --run-root "${OUTPUT_ROOT}/chunks/gpu0_can_fabric_fruit_jelly" \
  --run-root "${OUTPUT_ROOT}/chunks/gpu1_rice_vial_wallplugs" \
  --run-root "${OUTPUT_ROOT}/chunks/gpu2_walnuts_sheet_metal" \
  --output-json "${OUTPUT_ROOT}/fragmentation_analysis.json" \
  --output-tsv "${OUTPUT_ROOT}/fragmentation_analysis.tsv"

find "${OUTPUT_ROOT}" -type d -name anomaly_maps -prune -exec rm -rf {} +
printf 'run_name=%s\ncleanup_anomaly_maps=true\n' "${RUN_NAME}" \
  >"${OUTPUT_ROOT}/layerwise_remote_complete.txt"
