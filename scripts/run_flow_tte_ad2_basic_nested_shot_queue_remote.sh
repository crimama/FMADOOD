#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
QUEUE_NAME="${QUEUE_NAME:-flowtte_ad2_basic_nested_shots_1_2_4_8_20260714_v1}"
WAIT_PID="${WAIT_PID:-}"
SUPPORT_JSON="${SUPPORT_JSON:-${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json}"
QUEUE_ROOT="${RESULTS_ROOT}/${QUEUE_NAME}"
SHOT_LIST="${SHOT_LIST:-1 2 4 8}"
SHARD_LAYOUT="${SHARD_LAYOUT:-full4}"
REBASE_SUPPORT_TO_DATA_ROOT="${REBASE_SUPPORT_TO_DATA_ROOT:-0}"

mkdir -p "${QUEUE_ROOT}"
printf '%s\n' "queue_pid=$$" "wait_pid=${WAIT_PID}" >"${QUEUE_ROOT}/queue_contract.txt"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    state="$(ps -o stat= -p "${WAIT_PID}" 2>/dev/null | tr -d '[:space:]')"
    [[ "${state}" == Z* ]] && break
    sleep 30
  done
fi

for shot in ${SHOT_LIST//,/ }; do
  run_name="${QUEUE_NAME}_shot${shot}"
  run_root="${RESULTS_ROOT}/${run_name}"
  shot_support_json="${QUEUE_ROOT}/support_paths_shot${shot}.json"
  python3 - "${SUPPORT_JSON}" "${shot_support_json}" "${shot}" <<'PY'
import json
import sys
from pathlib import Path

source, destination, count = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
payload = json.loads(source.read_text(encoding="utf-8"))
trimmed = {name: paths[:count] for name, paths in payload.items()}
if any(len(paths) != count for paths in trimmed.values()):
    raise SystemExit(f"support prefix does not contain {count} paths for every object")
destination.write_text(json.dumps(trimmed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  if [[ -f "${run_root}/summary.json" ]]; then
    printf 'SKIP shot=%s reason=aggregate_exists\n' "${shot}" | tee -a "${QUEUE_ROOT}/queue.log"
    continue
  fi
  printf 'START shot=%s utc=%s\n' "${shot}" "$(date -u +%FT%TZ)" | tee -a "${QUEUE_ROOT}/queue.log"
  env RUN_NAME="${run_name}" DATA_ROOT="${DATA_ROOT}" PROJECT_ROOT=/workspace \
    FSAD_ROOT="${FSAD_ROOT}" OUTPUT_ROOT="${run_root}" SUPPORT_JSON="${shot_support_json}" \
    SHOTS="${shot}" RUN_SEED=0 SUPPORT_SELECTION="fixed_json=${shot_support_json}" \
    BACKBONE_MODEL=dinov2_vitl14 FEATURE_LAYERS=5,11,17,23 \
    RAW_BINARY_POSTPROCESS=closefill_erode GUIDED_IN_PLACE=0 \
    GPU_IDLE_MEMORY_MIB=512 SHARD_LAYOUT="${SHARD_LAYOUT}" \
    REBASE_SUPPORT_TO_DATA_ROOT="${REBASE_SUPPORT_TO_DATA_ROOT}" \
    LOO_STANDARDIZATION=off FMAD_DINOV2_OFFLINE=1 \
    bash "${FSAD_ROOT}/scripts/run_flow_tte_ad2_hplus_guided_morph_remote.sh"
  printf 'COMPLETE shot=%s utc=%s\n' "${shot}" "$(date -u +%FT%TZ)" | tee -a "${QUEUE_ROOT}/queue.log"
done

touch "${QUEUE_ROOT}/queue_complete.txt"
