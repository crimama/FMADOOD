#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-flowtte_hplus_dvt_superad_rotation8_all8_20260710_v1}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/results_remote/${RUN_NAME}}"

RUN_NAME="${RUN_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
BACKBONE_MODEL="${BACKBONE_MODEL:-dinov3_vith16plus}" \
FEATURE_LAYERS="${FEATURE_LAYERS:-7,15,23,31}" \
DVT_DENOISE_MODE="${DVT_DENOISE_MODE:-position_mean}" \
DVT_ALPHA="${DVT_ALPHA:-1.0}" \
NORMALITY_MODE="${NORMALITY_MODE:-fused}" \
SUPPORT_TRANSFORMS="${SUPPORT_TRANSFORMS:-superad_rot000,superad_rot045,superad_rot090,superad_rot135,superad_rot180,superad_rot225,superad_rot270,superad_rot315}" \
SUPPORT_BRIGHTNESS_RANGE="${SUPPORT_BRIGHTNESS_RANGE:-1.0,1.0}" \
CALIBRATION_SAMPLE_SIZE="${CALIBRATION_SAMPLE_SIZE:-4096}" \
CLEANUP_MAPS="${CLEANUP_MAPS:-1}" \
FMAD_DINOV3_OFFLINE="${FMAD_DINOV3_OFFLINE:-1}" \
bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"
