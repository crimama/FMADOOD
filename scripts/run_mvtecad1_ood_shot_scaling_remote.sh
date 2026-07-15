#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/home/woojun/dataset/mvtec_ad}"
OOD_PARENT="${OOD_PARENT:-/workspace/mvtecad_ood_data}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flowtte_mvtecad1_ood_shots_1_2_8_20260715_v1}"
RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
SHOTS="${SHOTS:-1 2 8}"
GPU_LAYOUT="${GPU_LAYOUT:-2}"
OBJECTS="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"

cd "${FSAD_ROOT}"
mkdir -p "${RUN_ROOT}/logs"
export FMAD_DINOV2_OFFLINE="${FMAD_DINOV2_OFFLINE:-1}"

run_condition() {
  local gpu="$1" shot="$2" condition="$3" root="$4"
  local output="${RUN_ROOT}/shot_${shot}/${condition}"
  local log="${RUN_ROOT}/logs/shot_${shot}_${condition}.log"
  if [[ -f "${output}/run_manifest.json" ]]; then
    echo "[skip] shot=${shot} condition=${condition}" | tee -a "${log}"
    return
  fi
  echo "[start] shot=${shot} condition=${condition} gpu=${gpu}" | tee "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/run_flow_tte_mvtec_ad1.py \
    --data-root "${root}" --output-root "${output}" \
    --project-root /workspace --fsad-root "${FSAD_ROOT}" \
    --objects "${OBJECTS}" --shots "${shot}" --seed 0 --device cuda \
    --backbone-model dinov2_vitb14_reg --preprocess-recipe fmad_shorter_edge \
    --image-size 448 --crop-size 448 --feature-layers 2,5,8,11 \
    --feature-fusion layer_norm_mean --support-selection first \
    --support-selection-seed 0 --support-transforms identity \
    --support-brightness-range 1.0,1.0 --flow-epochs 3 --coupling-layers 2 \
    --hidden-multiplier 1 --flow-lr 2e-4 --flow-clamp 1.9 \
    --flow-transform-mode flow --tail-weight 0.3 --tail-top-k-ratio 0.05 \
    --lambda-logdet 1e-3 --density-quantile 0.90 --expansion-budget 1.0 \
    --distance-weight 1.0 --density-weight 0.0 --score-mode latent_distance \
    --dvt-denoise-mode none --normality-mode fused --top-percent 0.01 \
    --query-chunk-size 512 --calibration-sample-size 0 \
    2>&1 | tee -a "${log}"
  echo "[complete] shot=${shot} condition=${condition}" | tee -a "${log}"
}

for shot in ${SHOTS//,/ }; do
  mkdir -p "${RUN_ROOT}/shot_${shot}"
  status=0
  if [[ "${GPU_LAYOUT}" == 4 ]]; then
    (run_condition 0 "${shot}" id "${DATA_ROOT}" && \
     run_condition 0 "${shot}" brightness "${OOD_PARENT}/mvtec_brightness_s3") & p0=$!
    run_condition 1 "${shot}" contrast "${OOD_PARENT}/mvtec_contrast_s3" & p1=$!
    run_condition 2 "${shot}" defocus_blur "${OOD_PARENT}/mvtec_defocus_blur_s3" & p2=$!
    run_condition 3 "${shot}" gaussian_noise "${OOD_PARENT}/mvtec_gaussian_noise_s3" & p3=$!
    wait "${p0}" || status=1
    wait "${p1}" || status=1
    wait "${p2}" || status=1
    wait "${p3}" || status=1
  else
    (run_condition 0 "${shot}" id "${DATA_ROOT}" && \
     run_condition 0 "${shot}" brightness "${OOD_PARENT}/mvtec_brightness_s3" && \
     run_condition 0 "${shot}" contrast "${OOD_PARENT}/mvtec_contrast_s3") & p0=$!
    (run_condition 1 "${shot}" defocus_blur "${OOD_PARENT}/mvtec_defocus_blur_s3" && \
     run_condition 1 "${shot}" gaussian_noise "${OOD_PARENT}/mvtec_gaussian_noise_s3") & p1=$!
    wait "${p0}" || status=1
    wait "${p1}" || status=1
  fi
  [[ "${status}" -eq 0 ]] || exit "${status}"
  python3 scripts/summarize_mvtecad1_ood_image_metrics.py \
    --run-root "${RUN_ROOT}/shot_${shot}" \
    --output "${RUN_ROOT}/shot_${shot}/summary.json" \
    >"${RUN_ROOT}/logs/shot_${shot}_summary.log" 2>&1
  echo "COMPLETE shot=${shot} utc=$(date -u +%FT%TZ)" | tee -a "${RUN_ROOT}/queue.log"
done

touch "${RUN_ROOT}/remote_run_complete.txt"
