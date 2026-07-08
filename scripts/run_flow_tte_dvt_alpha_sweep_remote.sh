#!/usr/bin/env bash
set -euo pipefail

ALPHAS="${ALPHAS:-0.25 0.5 0.75}"
RUN_SUFFIX="${RUN_SUFFIX:-20260708_v1}"
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"

for alpha in ${ALPHAS}; do
  alpha_tag="$(printf '%s' "${alpha}" | tr -d '.')"
  run_name="flowtte_dvt_denoising_all8_a${alpha_tag}_${RUN_SUFFIX}"
  complete_path="${RESULTS_ROOT}/${run_name}/remote_run_complete.txt"
  if [[ -f "${complete_path}" ]]; then
    printf 'skip_existing=%s\n' "${run_name}"
    continue
  fi
  printf 'run_alpha=%s run_name=%s\n' "${alpha}" "${run_name}"
  RUN_NAME="${run_name}" DVT_ALPHA="${alpha}" \
    bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"
done
