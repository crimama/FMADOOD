#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
STRUCTURE_MODE="${STRUCTURE_MODE:?STRUCTURE_MODE is required}"

DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"
BACKBONE_MODEL="${BACKBONE_MODEL:-dinov3_vitl16}"
SUPPORT_SELECTION="${SUPPORT_SELECTION:-dinov3_cls_greedy_coreset}"
CONTEXT_TOP_M="${CONTEXT_TOP_M:-4}"

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
  --density-weight 0.25
  --score-mode latent_distance
  --top-percent 0.01
  --query-chunk-size 512
  --pro-integration-limit 0.05
  --rgb-guide none
  --cleanup-maps
  --backbone-model "${BACKBONE_MODEL}"
  --support-selection "${SUPPORT_SELECTION}"
  --support-transforms identity
  --feature-fusion layer_norm_mean
  --context-source register
)

case "${STRUCTURE_MODE}" in
  register_top_m)
    common_args+=(--context-mode top_m --context-top-m "${CONTEXT_TOP_M}")
    ;;
  register_conditional_nf)
    common_args+=(--context-mode none --flow-condition-mode context)
    ;;
  *)
    echo "Unsupported STRUCTURE_MODE: ${STRUCTURE_MODE}" >&2
    exit 2
    ;;
esac

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

printf 'run_name=%s\nstructure_mode=%s\ncontext_top_m=%s\n' \
  "${RUN_NAME}" "${STRUCTURE_MODE}" "${CONTEXT_TOP_M}" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
