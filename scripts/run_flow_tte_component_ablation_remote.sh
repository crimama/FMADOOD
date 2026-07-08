#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_component_ablation_all8_20260707_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"
BACKBONE_MODEL="${BACKBONE_MODEL:-dinov3_vitl16}"
SUPPORT_SELECTION="${SUPPORT_SELECTION:-fixed_json=${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json}"
OBJECTS="${OBJECTS:-can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal}"

export FMAD_DINOV3_OFFLINE="${FMAD_DINOV3_OFFLINE:-1}"

mkdir -p "${OUTPUT_ROOT}/variants" "${OUTPUT_ROOT}/logs"

common_args=(
  --data-root "${DATA_ROOT}"
  --project-root "${PROJECT_ROOT}"
  --fsad-root "${FSAD_ROOT}"
  --objects "${OBJECTS}"
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
  --support-selection "${SUPPORT_SELECTION}"
  --support-transforms identity
  --feature-fusion layer_norm_mean
)

run_variant() {
  local cuda_slot="$1"
  local variant="$2"
  shift 2
  local variant_root="${OUTPUT_ROOT}/variants/${variant}"
  local log_path="${OUTPUT_ROOT}/logs/${variant}.log"

  CUDA_VISIBLE_DEVICES="${cuda_slot}" python3 "${FSAD_ROOT}/scripts/run_flow_tte_mvtec_ad2.py" \
    "${common_args[@]}" \
    --output-root "${variant_root}" \
    "$@" \
    >"${log_path}" 2>&1
}

run_variant 0 identity_no_nf_dist \
  --flow-transform-mode identity \
  --density-weight 0.0 \
  --context-source none \
  --context-mode none \
  --flow-condition-mode none &
pid0=$!

run_variant 1 flow_dist_only \
  --density-weight 0.0 \
  --context-source none \
  --context-mode none \
  --flow-condition-mode none &
pid1=$!

run_variant 2 flow_tte_budget125 \
  --expansion-budget 1.25 \
  --density-weight 0.25 \
  --context-source none \
  --context-mode none \
  --flow-condition-mode none &
pid2=$!

wait "${pid0}"
wait "${pid1}"
wait "${pid2}"

run_variant 0 cls_topm4 \
  --context-source cls \
  --context-mode top_m \
  --context-top-m 4 \
  --flow-condition-mode none &
pid0=$!

run_variant 1 register_condnf \
  --context-source register \
  --context-mode none \
  --flow-condition-mode context &
pid1=$!

run_variant 2 hybrid_regcond_cls_topm4 \
  --context-source cls \
  --flow-context-source register \
  --memory-context-source cls \
  --flow-condition-mode context \
  --context-mode top_m \
  --context-top-m 4 &
pid2=$!

wait "${pid0}"
wait "${pid1}"
wait "${pid2}"

printf 'run_name=%s\nobjects=%s\nbackbone=%s\nsupport_selection=%s\n' \
  "${RUN_NAME}" "${OBJECTS}" "${BACKBONE_MODEL}" "${SUPPORT_SELECTION}" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
