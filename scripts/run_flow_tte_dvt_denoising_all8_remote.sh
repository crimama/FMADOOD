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
CLEANUP_MAPS="${CLEANUP_MAPS:-1}"
FLOW_TRANSFORM_MODE="${FLOW_TRANSFORM_MODE:-flow}"
DENSITY_WEIGHT="${DENSITY_WEIGHT:-0.25}"
SCORE_MODE="${SCORE_MODE:-latent_distance}"

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
  --top-percent 0.01
  --query-chunk-size 512
  --pro-integration-limit 0.05
  --backbone-model "${BACKBONE_MODEL}"
  --feature-layers "${FEATURE_LAYERS}"
  --support-selection "${SUPPORT_SELECTION}"
  --support-transforms identity
  --feature-fusion layer_norm_mean
  --context-source none
  --context-mode none
  --flow-condition-mode none
  --flow-transform-mode "${FLOW_TRANSFORM_MODE}"
  --dvt-denoise-mode position_mean
  --dvt-denoise-alpha "${DVT_ALPHA}"
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

printf 'run_name=%s\nbackbone=%s\nfeature_layers=%s\nsupport_selection=%s\ndvt_alpha=%s\nflow_transform_mode=%s\ndensity_weight=%s\nscore_mode=%s\n' \
  "${RUN_NAME}" "${BACKBONE_MODEL}" "${FEATURE_LAYERS}" "${SUPPORT_SELECTION}" "${DVT_ALPHA}" \
  "${FLOW_TRANSFORM_MODE}" "${DENSITY_WEIGHT}" "${SCORE_MODE}" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
