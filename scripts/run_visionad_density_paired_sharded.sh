#!/usr/bin/env bash
set -euo pipefail
FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
HOST_TAG="${HOST_TAG:?HOST_TAG is required}"
ROOT="${RESULTS_ROOT}/${RUN_NAME}"
EVALUATE_EXISTING="${EVALUATE_EXISTING:-0}"
ALL="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"
FIRST="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut"
LAST="pill,screw,tile,toothbrush,transistor,wood,zipper"
mkdir -p "${ROOT}/logs"; cd "${FSAD_ROOT}"
run_part() {
  local gpu="$1" shot="$2" part="$3" objects="$4" out
  out="${ROOT}/shot_${shot}_${part}"
  test -f "${out}/baseline/run_manifest.json" -a -f "${out}/pseudo_density/run_manifest.json" && return 0
  local eval_args=(); test "${EVALUATE_EXISTING}" = 1 && eval_args+=(--evaluate-existing)
  CUDA_VISIBLE_DEVICES="${gpu}" FMAD_DINOV2_OFFLINE=1 python3 -m visionad_density.run_mvtec \
    --data-root "${DATA_ROOT}" --output-root "${out}" --objects "${objects}" \
    --shots "${shot}" --seed 0 --device cuda --density-weight 0.25 "${eval_args[@]}" \
    >"${ROOT}/logs/shot_${shot}_${part}.log" 2>&1
}
pids=()
case "${HOST_TAG}" in
  dsba3)
    run_part 0 1 all "${ALL}" & pids+=("$!")
    run_part 1 2 all "${ALL}" & pids+=("$!")
    run_part 2 4 all "${ALL}" & pids+=("$!")
    run_part 3 8 all "${ALL}" & pids+=("$!") ;;
  dsba5)
    run_part 0 16 first "${FIRST}" & pids+=("$!")
    run_part 1 16 last "${LAST}" & pids+=("$!") ;;
  *) exit 2 ;;
esac
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
test "${status}" -eq 0; date -u +%FT%TZ >"${ROOT}/${HOST_TAG}_complete.txt"
