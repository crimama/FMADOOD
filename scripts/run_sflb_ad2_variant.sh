#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
SUPPORT_JSON="${SUPPORT_JSON:?SUPPORT_JSON is required}"
GPU_SLOT="${GPU_SLOT:?GPU_SLOT is required}"
SHIFT_RANK="${SHIFT_RANK:?SHIFT_RANK is required}"
SHIFT_TRIM="${SHIFT_TRIM:-0.20}"
SHIFT_STRENGTH="${SHIFT_STRENGTH:-1.0}"

mkdir -p "${OUTPUT_ROOT}"
env CUDA_VISIBLE_DEVICES="${GPU_SLOT}" FMAD_DINOV2_OFFLINE=1 \
  python3 "${FSAD_ROOT}/scripts/run_flow_tte_mvtec_ad2.py" \
    --data-root "${DATA_ROOT}" \
    --project-root /workspace \
    --fsad-root "${FSAD_ROOT}" \
    --shots 16 --seed 0 --device cuda \
    --flow-epochs 3 --coupling-layers 2 --hidden-multiplier 1 \
    --flow-lr 2e-4 --flow-clamp 1.9 --tail-weight 0.3 --tail-top-k-ratio 0.05 \
    --lambda-logdet 1e-3 --density-quantile 0.90 --expansion-budget 1.0 \
    --distance-weight 1.0 --density-weight 0.25 --score-mode latent_distance \
    --top-percent 0.01 --query-chunk-size 512 --pro-integration-limit 0.05 \
    --rgb-guide guided_r8 --binary-postprocess closefill_erode \
    --morphology-line-length 17 --morphology-angle-count 16 \
    --backbone-model dinov2_vitl14 --feature-layers 5,11,17,23 \
    --support-selection "fixed_json=${SUPPORT_JSON}" --support-selection-seed 0 \
    --support-transforms identity --feature-fusion layer_norm_mean \
    --normality-mode fused --dvt-denoise-mode position_mean --dvt-denoise-alpha 1.0 \
    --shift-projection-rank "${SHIFT_RANK}" \
    --shift-projection-trim "${SHIFT_TRIM}" \
    --shift-projection-max-samples 32768 \
    --shift-projection-strength "${SHIFT_STRENGTH}" \
    --output-root "${OUTPUT_ROOT}" \
    --objects can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal
