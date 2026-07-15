#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/MVTecAD}"
OOD_PARENT="${OOD_PARENT:-/home/hunim/Volume/DATA/mvtecad_ood_data}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flowtte_mvtecad1_ood_shots_2_8_20260715_v1}"
RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"

mkdir -p "${RUN_ROOT}/logs" "${OOD_PARENT}"
cd "${FSAD_ROOT}"
python3 -m pip install --quiet --disable-pip-version-check imagecorruptions==1.1.2

prepare_one() {
  local corruption="$1"
  python3 scripts/prepare_mvtecad1_ood.py \
    --source-root "${DATA_ROOT}" --output-parent "${OOD_PARENT}" --severity 3 \
    --corruptions "${corruption}" \
    >"${RUN_ROOT}/logs/prepare_${corruption}.log" 2>&1
}

prepare_one brightness & p0=$!
prepare_one contrast & p1=$!
prepare_one defocus_blur & p2=$!
prepare_one gaussian_noise & p3=$!
status=0
wait "${p0}" || status=1
wait "${p1}" || status=1
wait "${p2}" || status=1
wait "${p3}" || status=1
[[ "${status}" -eq 0 ]] || exit "${status}"

env FSAD_ROOT="${FSAD_ROOT}" DATA_ROOT="${DATA_ROOT}" OOD_PARENT="${OOD_PARENT}" \
  RESULTS_ROOT="${RESULTS_ROOT}" RUN_NAME="${RUN_NAME}" SHOTS="2 8" GPU_LAYOUT=4 \
  bash scripts/run_mvtecad1_ood_shot_scaling_remote.sh
