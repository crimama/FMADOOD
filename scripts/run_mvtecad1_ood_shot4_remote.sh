#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/home/woojun/dataset/mvtec_ad}"
OOD_PARENT="${OOD_PARENT:-/workspace/mvtecad_ood_data}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flowtte_mvtecad1_ood_shot4_20260714_v1}"
RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
OBJECTS="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"

cd "${FSAD_ROOT}"
mkdir -p "${RUN_ROOT}/logs"
export FMAD_DINOV2_OFFLINE="${FMAD_DINOV2_OFFLINE:-1}"

python3 -m pip install --quiet --disable-pip-version-check imagecorruptions==1.1.2
python3 scripts/prepare_mvtecad1_ood.py \
  --source-root "${DATA_ROOT}" --output-parent "${OOD_PARENT}" --severity 3 \
  >"${RUN_ROOT}/logs/prepare.log" 2>&1

run_condition() {
  local gpu="$1"
  local condition="$2"
  local root="$3"
  local output="${RUN_ROOT}/${condition}"
  local log="${RUN_ROOT}/logs/${condition}.log"
  if [[ -f "${output}/run_manifest.json" ]]; then
    echo "[skip] ${condition}" | tee -a "${log}"
    return
  fi
  echo "[start] condition=${condition} gpu=${gpu}" | tee "${log}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/run_flow_tte_mvtec_ad1.py \
    --data-root "${root}" --output-root "${output}" \
    --project-root /workspace --fsad-root "${FSAD_ROOT}" \
    --objects "${OBJECTS}" --shots 4 --seed 0 --device cuda \
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
  echo "[complete] condition=${condition}" | tee -a "${log}"
}

(run_condition 0 id "${DATA_ROOT}" && \
 run_condition 0 brightness "${OOD_PARENT}/mvtec_brightness_s3" && \
 run_condition 0 contrast "${OOD_PARENT}/mvtec_contrast_s3") & p0=$!
(run_condition 1 defocus_blur "${OOD_PARENT}/mvtec_defocus_blur_s3" && \
 run_condition 1 gaussian_noise "${OOD_PARENT}/mvtec_gaussian_noise_s3") & p1=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
[[ "${status}" -eq 0 ]] || exit "${status}"

python3 scripts/summarize_mvtecad1_ood_image_metrics.py \
  --run-root "${RUN_ROOT}" --output "${RUN_ROOT}/summary.json" \
  >"${RUN_ROOT}/logs/summary.log" 2>&1
touch "${RUN_ROOT}/remote_run_complete.txt"
