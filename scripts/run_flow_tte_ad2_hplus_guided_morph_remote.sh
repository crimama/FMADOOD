#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_ad2_hplus_guided_r8_morph_all8_20260713_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"
SUPPORT_JSON="${SUPPORT_JSON:-${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json}"
SHOTS="${SHOTS:-16}"
RUN_SEED="${RUN_SEED:-0}"
SUPPORT_SELECTION="${SUPPORT_SELECTION:-fixed_json=${SUPPORT_JSON}}"
SUPPORT_SELECTION_SEED="${SUPPORT_SELECTION_SEED:-${RUN_SEED}}"
BACKBONE_MODEL="${BACKBONE_MODEL:-dinov3_vith16plus}"
FEATURE_LAYERS="${FEATURE_LAYERS:-7,15,23,31}"
RAW_BINARY_POSTPROCESS="${RAW_BINARY_POSTPROCESS:-closefill_erode}"
GUIDED_IN_PLACE="${GUIDED_IN_PLACE:-0}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-512}"
GPU_IDLE_POLL_SECONDS="${GPU_IDLE_POLL_SECONDS:-30}"
SHARD_LAYOUT="${SHARD_LAYOUT:-full4}"
REBASE_SUPPORT_TO_DATA_ROOT="${REBASE_SUPPORT_TO_DATA_ROOT:-0}"
LOO_STANDARDIZATION="${LOO_STANDARDIZATION:-on}"

export FMAD_DINOV3_OFFLINE="${FMAD_DINOV3_OFFLINE:-1}"
mkdir -p "${OUTPUT_ROOT}/chunks" "${OUTPUT_ROOT}/logs"

[[ -d "${DATA_ROOT}" ]] || { echo "missing data root: ${DATA_ROOT}" >&2; exit 1; }
if [[ "${SUPPORT_SELECTION}" == fixed_json=* ]]; then
  [[ -f "${SUPPORT_JSON}" ]] || { echo "missing fixed support JSON: ${SUPPORT_JSON}" >&2; exit 1; }
fi

if [[ "${REBASE_SUPPORT_TO_DATA_ROOT}" == "1" ]]; then
  case "${SHARD_LAYOUT}" in
    full4|full2) support_objects="can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal" ;;
    first4_2gpu) support_objects="can,fabric,fruit_jelly,rice" ;;
    first4_4gpu) support_objects="can,fabric,fruit_jelly,rice" ;;
    last4_4gpu) support_objects="vial,wallplugs,walnuts,sheet_metal" ;;
    *) echo "unknown SHARD_LAYOUT: ${SHARD_LAYOUT}" >&2; exit 2 ;;
  esac
  rebased_support_json="${OUTPUT_ROOT}/fixed_support_paths_rebased.json"
  python3 "${FSAD_ROOT}/scripts/rebase_fixed_support_json.py" \
    --input "${SUPPORT_JSON}" \
    --data-root "${DATA_ROOT}" \
    --objects "${support_objects}" \
    --output "${rebased_support_json}"
  SUPPORT_JSON="${rebased_support_json}"
  SUPPORT_SELECTION="fixed_json=${SUPPORT_JSON}"
fi

case "${SHARD_LAYOUT}" in
  full4)
    shard_specs=(
      "0|gpu0_can_fabric|can,fabric"
      "1|gpu1_fruit_jelly_rice|fruit_jelly,rice"
      "2|gpu2_vial_wallplugs|vial,wallplugs"
      "3|gpu3_walnuts_sheet_metal|walnuts,sheet_metal"
    )
    required_gpu_count=4
    ;;
  full2)
    shard_specs=(
      "0|gpu0_can_fabric_fruit_jelly_rice|can,fabric,fruit_jelly,rice"
      "1|gpu1_vial_wallplugs_walnuts_sheet_metal|vial,wallplugs,walnuts,sheet_metal"
    )
    required_gpu_count=2
    ;;
  first4_2gpu)
    shard_specs=(
      "0|gpu0_can_fabric|can,fabric"
      "1|gpu1_fruit_jelly_rice|fruit_jelly,rice"
    )
    required_gpu_count=2
    ;;
  first4_4gpu)
    shard_specs=(
      "0|gpu0_can|can"
      "1|gpu1_fabric|fabric"
      "2|gpu2_fruit_jelly|fruit_jelly"
      "3|gpu3_rice|rice"
    )
    required_gpu_count=4
    ;;
  last4_4gpu)
    shard_specs=(
      "0|gpu0_vial|vial"
      "1|gpu1_wallplugs|wallplugs"
      "2|gpu2_walnuts|walnuts"
      "3|gpu3_sheet_metal|sheet_metal"
    )
    required_gpu_count=4
    ;;
  *)
    echo "unknown SHARD_LAYOUT: ${SHARD_LAYOUT}" >&2
    exit 2
    ;;
esac

if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt "${required_gpu_count}" ]]; then
  echo "${required_gpu_count} visible GPUs are required for ${SHARD_LAYOUT}" >&2
  exit 1
fi

common_args=(
  --data-root "${DATA_ROOT}"
  --project-root "${PROJECT_ROOT}"
  --fsad-root "${FSAD_ROOT}"
  --shots "${SHOTS}"
  --seed "${RUN_SEED}"
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
  --residual-weight 0.25
  --top-percent 0.01
  --query-chunk-size 512
  --pro-integration-limit 0.05
  --rgb-guide none
  --binary-postprocess "${RAW_BINARY_POSTPROCESS}"
  --morphology-line-length 17
  --morphology-angle-count 16
  --backbone-model "${BACKBONE_MODEL}"
  --backbone-resolution 0
  --feature-layers "${FEATURE_LAYERS}"
  --tile-patch-size 0
  --tile-overlap 0
  --image-resize-factor 1.0
  --support-brightness-range 1.0,1.0
  --support-selection "${SUPPORT_SELECTION}"
  --support-selection-seed "${SUPPORT_SELECTION_SEED}"
  --support-transforms identity
  --feature-fusion layer_norm_mean
  --normality-mode fused
  --context-source none
  --flow-context-source auto
  --memory-context-source auto
  --context-mode none
  --context-weight 0.0
  --context-top-m 1
  --calibration-sample-size 0
  --flow-condition-mode none
  --transformer-context-mode none
  --flow-transform-mode flow
  --loo-standardization "${LOO_STANDARDIZATION}"
  --dvt-denoise-mode position_mean
  --dvt-denoise-alpha 1.0
  --score-field-calibration-mode none
  --score-field-calibration-alpha 1.0
  --score-field-position-std-floor 0.25
  --score-field-foreground-mode none
  --score-field-foreground-quantile 0.20
  --score-field-background-multiplier 0.50
  --score-field-foreground-smooth-kernel 5
  --score-field-support-score-quantile 0.90
)

wait_for_gpu() {
  local cuda_slot="$1"
  local used_mib
  while true; do
    used_mib="$(
      nvidia-smi --id="${cuda_slot}" --query-gpu=memory.used --format=csv,noheader,nounits \
        | tr -d '[:space:]'
    )"
    if [[ "${used_mib}" =~ ^[0-9]+$ ]] && [[ "${used_mib}" -le "${GPU_IDLE_MEMORY_MIB}" ]]; then
      printf 'gpu=%s idle memory_used_mib=%s\n' "${cuda_slot}" "${used_mib}"
      return 0
    fi
    printf 'gpu=%s waiting memory_used_mib=%s threshold_mib=%s\n' \
      "${cuda_slot}" "${used_mib}" "${GPU_IDLE_MEMORY_MIB}"
    sleep "${GPU_IDLE_POLL_SECONDS}"
  done
}

run_shard() {
  local cuda_slot="$1"
  local shard_name="$2"
  local objects="$3"
  local shard_root="${OUTPUT_ROOT}/chunks/${shard_name}"
  local raw_root="${shard_root}/raw"
  local guided_root="${shard_root}/guided_r8_morph"

  wait_for_gpu "${cuda_slot}" >>"${OUTPUT_ROOT}/logs/${shard_name}_queue.log" 2>&1

  env CUDA_VISIBLE_DEVICES="${cuda_slot}" \
    python3 "${FSAD_ROOT}/scripts/run_flow_tte_mvtec_ad2.py" \
      "${common_args[@]}" \
      --output-root "${raw_root}" \
      --objects "${objects}" \
      >"${OUTPUT_ROOT}/logs/${shard_name}_raw.log" 2>&1

  if [[ "${GUIDED_IN_PLACE}" == "1" ]]; then
    cp "${raw_root}/run_manifest.json" "${raw_root}/upstream_run_manifest.json"
    cp "${raw_root}/metrics.json" "${raw_root}/upstream_metrics.json"
    cp "${raw_root}/metrics_seed=${RUN_SEED}.json" \
      "${raw_root}/upstream_metrics_seed=${RUN_SEED}.json"
    env CUDA_VISIBLE_DEVICES="${cuda_slot}" \
      python3 "${FSAD_ROOT}/scripts/run_flow_tte_mvtecad2_guided_refinement.py" \
        --data-root "${DATA_ROOT}" \
        --source-root "${raw_root}" \
        --output-root "${raw_root}" \
        --objects "${objects}" \
        --seed "${RUN_SEED}" \
        --binary-postprocess closefill_erode \
        --morphology-line-length 17 \
        --morphology-angle-count 16 \
        --cleanup-source-maps \
        --cleanup-output-maps \
        >"${OUTPUT_ROOT}/logs/${shard_name}_guided.log" 2>&1
    mkdir -p "${guided_root}"
    cp "${raw_root}/run_manifest.json" "${guided_root}/run_manifest.json"
    cp "${raw_root}/metrics.json" "${guided_root}/metrics.json"
    cp "${raw_root}/metrics_seed=${RUN_SEED}.json" \
      "${guided_root}/metrics_seed=${RUN_SEED}.json"
    cp "${raw_root}/cleanup_evidence.txt" "${guided_root}/cleanup_evidence.txt"
    mv "${raw_root}/upstream_run_manifest.json" "${raw_root}/run_manifest.json"
    mv "${raw_root}/upstream_metrics.json" "${raw_root}/metrics.json"
    mv "${raw_root}/upstream_metrics_seed=${RUN_SEED}.json" \
      "${raw_root}/metrics_seed=${RUN_SEED}.json"
  else
    env CUDA_VISIBLE_DEVICES="${cuda_slot}" \
      python3 "${FSAD_ROOT}/scripts/run_flow_tte_mvtecad2_guided_refinement.py" \
        --data-root "${DATA_ROOT}" \
        --source-root "${raw_root}" \
        --output-root "${guided_root}" \
        --objects "${objects}" \
        --seed "${RUN_SEED}" \
        --binary-postprocess closefill_erode \
        --morphology-line-length 17 \
        --morphology-angle-count 16 \
        --cleanup-source-maps \
        --cleanup-output-maps \
        >"${OUTPUT_ROOT}/logs/${shard_name}_guided.log" 2>&1
  fi
}

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

for spec in "${shard_specs[@]}"; do
  IFS='|' read -r cuda_slot shard_name objects <<<"${spec}"
  run_shard "${cuda_slot}" "${shard_name}" "${objects}" & child_pids+=("$!")
done

status=0
for pid in "${child_pids[@]}"; do
  wait "${pid}" || status=$?
done
trap - EXIT INT TERM
[[ "${status}" -eq 0 ]] || exit "${status}"

if [[ "${SHARD_LAYOUT}" == "full4" || "${SHARD_LAYOUT}" == "full2" ]]; then
  python3 "${FSAD_ROOT}/scripts/aggregate_flow_tte_ad2_guided_chunks.py" \
    --root "${OUTPUT_ROOT}" >"${OUTPUT_ROOT}/logs/aggregate.log" 2>&1
fi

method_bundle_sha256="$(
  sha256sum \
    "${FSAD_ROOT}/scripts/run_flow_tte_mvtec_ad2.py" \
    "${FSAD_ROOT}/scripts/run_flow_tte_mvtecad2_guided_refinement.py" \
    "${FSAD_ROOT}/scripts/run_flow_tte_ad2_hplus_guided_morph_remote.sh" \
    "${FSAD_ROOT}/src/flow_tte_phase2_refinement.py" \
    "${PROJECT_ROOT}/fmad/evaluation/metrics.py" \
    | sha256sum | awk '{print $1}'
)"
printf '%s\n' \
  "run_name=${RUN_NAME}" \
  "shard_layout=${SHARD_LAYOUT}" \
  "method_bundle_sha256=${method_bundle_sha256}" \
  "backbone=${BACKBONE_MODEL}" \
  "feature_layers=${FEATURE_LAYERS}" \
  "resolution=dataset_default_672_except_sheet_metal_448" \
  "shots=${SHOTS}" \
  "run_seed=${RUN_SEED}" \
  "support_selection=${SUPPORT_SELECTION}" \
  "support_selection_seed=${SUPPORT_SELECTION_SEED}" \
  "raw_binary_postprocess=${RAW_BINARY_POSTPROCESS}" \
  "guided_in_place=${GUIDED_IN_PLACE}" \
  "dvt=position_mean_alpha1.0" \
  "density_weight=0.25" \
  "loo_standardization=${LOO_STANDARDIZATION}" \
  "rgb_guide=guided_r8_eps1e-2_half_scale" \
  "binary_postprocess=closefill_erode_line17_angles16" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
