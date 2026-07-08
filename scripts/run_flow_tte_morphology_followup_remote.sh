#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_morphology_reduced_20260707_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"
BACKBONE_MODEL="${BACKBONE_MODEL:-dinov3_vitl16}"
SUPPORT_SELECTION="${SUPPORT_SELECTION:-fixed_json=${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json}"
OBJECTS="${OBJECTS:-fabric,can,wallplugs,vial}"

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

run_variant 0 no_context \
  --context-source none \
  --context-mode none \
  --flow-condition-mode none &
pid0=$!

run_variant 1 cls_w10 \
  --context-source cls \
  --context-mode soft_penalty \
  --context-weight 10 \
  --flow-condition-mode none &
pid1=$!

run_variant 2 register_topm4 \
  --context-source register \
  --context-mode top_m \
  --context-top-m 4 \
  --flow-condition-mode none &
pid2=$!

wait "${pid0}"
wait "${pid1}"
wait "${pid2}"

run_variant 0 register_condnf \
  --context-source register \
  --context-mode none \
  --flow-condition-mode context

python3 "${FSAD_ROOT}/scripts/flow_tte_morphology_audit.py" \
  --data-root "${DATA_ROOT}" \
  --objects "${OBJECTS}" \
  --run-root "no_context=${OUTPUT_ROOT}/variants/no_context" \
  --run-root "cls_w10=${OUTPUT_ROOT}/variants/cls_w10" \
  --run-root "register_topm4=${OUTPUT_ROOT}/variants/register_topm4" \
  --run-root "register_condnf=${OUTPUT_ROOT}/variants/register_condnf" \
  --output-tsv "${OUTPUT_ROOT}/morphology_audit.tsv" \
  --output-json "${OUTPUT_ROOT}/morphology_audit.json"

printf 'run_name=%s\nobjects=%s\nbackbone=%s\nsupport_selection=%s\n' \
  "${RUN_NAME}" "${OBJECTS}" "${BACKBONE_MODEL}" "${SUPPORT_SELECTION}" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
