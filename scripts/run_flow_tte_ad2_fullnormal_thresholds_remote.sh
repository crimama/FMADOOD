#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_ad2_fullnormal_superadd_thresholds_20260713_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"
GPU0_OBJECTS="${GPU0_OBJECTS:-}"
GPU1_OBJECTS="${GPU1_OBJECTS:-}"
GPU2_OBJECTS="${GPU2_OBJECTS:-}"
GPU3_OBJECTS="${GPU3_OBJECTS:-}"

export FMAD_DINOV3_OFFLINE=1
mkdir -p "${OUTPUT_ROOT}/objects" "${OUTPUT_ROOT}/logs"

common_args=(
  --data-root "${DATA_ROOT}"
  --project-root "${PROJECT_ROOT}"
  --fsad-root "${FSAD_ROOT}"
  --shots 0
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
  --residual-weight 0.25
  --top-percent 0.01
  --query-chunk-size 512
  --pro-integration-limit 0.05
  --rgb-guide guided_r8
  --threshold-calibration-mode superadd_train95
  --threshold-fraction 8
  --threshold-percentile 95
  --threshold-factor 1.421
  --binary-postprocess closefill_erode
  --morphology-line-length 17
  --morphology-angle-count 16
  --backbone-model dinov3_vith16plus
  --backbone-resolution 0
  --feature-layers 7,15,23,31
  --tile-patch-size 0
  --tile-overlap 0
  --image-resize-factor 1.0
  --support-brightness-range 1.0,1.0
  --support-selection superadd_full_7of8
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
  --latent-bank-subsample superadd_knn_score
  --latent-bank-target-count 100000
  --flow-condition-mode none
  --transformer-context-mode none
  --flow-transform-mode flow
  --dvt-denoise-mode position_mean
  --dvt-denoise-alpha 1.0
  --score-field-calibration-mode none
  --cleanup-maps
)

run_gpu_queue() {
  local cuda_slot="$1"
  local object_list="$2"
  local object_name
  [[ -n "${object_list}" ]] || return 0
  for object_name in ${object_list//,/ }; do
    local object_root="${OUTPUT_ROOT}/objects/${object_name}"
    [[ -d "${DATA_ROOT}/${object_name}/train/good" ]] || {
      echo "missing train/good for ${object_name}" >&2
      return 1
    }
    env CUDA_VISIBLE_DEVICES="${cuda_slot}" \
      python3 "${FSAD_ROOT}/scripts/run_flow_tte_mvtec_ad2.py" \
        "${common_args[@]}" \
        --objects "${object_name}" \
        --output-root "${object_root}" \
        >"${OUTPUT_ROOT}/logs/${object_name}.log" 2>&1
  done
}

pids=()
for slot in 0 1 2 3; do
  variable="GPU${slot}_OBJECTS"
  object_list="${!variable}"
  if [[ -n "${object_list}" ]]; then
    run_gpu_queue "${slot}" "${object_list}" & pids+=("$!")
  fi
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
[[ "${status}" -eq 0 ]] || exit "${status}"

printf '%s\n' \
  "run_name=${RUN_NAME}" \
  "support=sorted_train_good_7of8" \
  "threshold=heldout_1of8_p95_x1.421" \
  "oracle=test_public_raw_best_threshold" \
  "rgb_guide=guided_r8_eps1e-2_half_scale" \
  "morphology=closefill_erode_line17_angles16" \
  >"${OUTPUT_ROOT}/remote_run_complete.txt"
