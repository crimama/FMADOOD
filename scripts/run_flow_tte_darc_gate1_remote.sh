#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
RUN_ROOT="${2:-/workspace/results_remote/darc_gate1_20260710}"
DATA_ROOT="${DARC_AD1_ROOT:-/home/hunim/Volume/DATA/MVTecAD}"
SCRIPT="/workspace/scripts/run_flow_tte_darc_gate1.py"

if [[ "${MODE}" != "full" && "${MODE}" != "smoke" ]]; then
  echo "usage: $0 [full|smoke] [run-root]" >&2
  exit 2
fi
if [[ ! -f "${SCRIPT}" || ! -d "${DATA_ROOT}" ]]; then
  echo "missing runner or AD1 root: ${SCRIPT} ${DATA_ROOT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs"
exec 9>"${RUN_ROOT}/controller.lock"
if ! flock -n 9; then
  echo "DARC Gate1 controller already owns ${RUN_ROOT}" >&2
  exit 3
fi

{
  date --iso-8601=seconds
  hostname
  pwd
  df -h /workspace
  nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv,noheader
} >"${RUN_ROOT}/preflight.txt"

run_lane() {
  local gpu="$1"
  shift
  local object_name
  for object_name in "$@"; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
      python3 "${SCRIPT}" \
        --data-root "${DATA_ROOT}" \
        --output-root "${RUN_ROOT}" \
        --object "${object_name}" \
        --device cuda:0 \
        --seeds 0,1,2 \
        --resume \
      >"${RUN_ROOT}/logs/gpu${gpu}_${object_name}.log" 2>&1
  done
}

if [[ "${MODE}" == "smoke" ]]; then
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
    python3 "${SCRIPT}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${RUN_ROOT}" \
      --object bottle \
      --device cuda:0 \
      --seeds 0 \
      --smoke \
      --resume \
    >"${RUN_ROOT}/logs/gpu0_bottle_smoke.log" 2>&1
  date --iso-8601=seconds >"${RUN_ROOT}/smoke_complete.txt"
  exit 0
fi

run_lane 0 bottle carpet leather screw transistor &
pid0=$!
run_lane 1 cable grid metal_nut tile wood &
pid1=$!
run_lane 2 capsule hazelnut pill toothbrush zipper &
pid2=$!
trap 'kill "${pid0}" "${pid1}" "${pid2}" 2>/dev/null || true' INT TERM

set +e
wait "${pid0}"; status0=$?
wait "${pid1}"; status1=$?
wait "${pid2}"; status2=$?
set -e
if (( status0 != 0 || status1 != 0 || status2 != 0 )); then
  echo "lane failure: gpu0=${status0} gpu1=${status1} gpu2=${status2}" >&2
  exit 4
fi

python3 - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
complete = sorted(root.glob("objects/*/seed=*/complete.json"))
if len(complete) != 45:
    raise SystemExit(f"expected 45 completion files, found {len(complete)}")
hashes = set()
for path in complete:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("smoke") is not False or payload.get("source_count") != 16:
        raise SystemExit(f"invalid reportable completion: {path}")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise SystemExit(f"missing completion provenance: {path}")
    hashes.add(provenance.get("method_sha256"))
if len(hashes) != 1:
    raise SystemExit(f"method hash mismatch: {hashes}")
PY

python3 "${SCRIPT}" --output-root "${RUN_ROOT}" --aggregate \
  >"${RUN_ROOT}/logs/aggregate.log" 2>&1
date --iso-8601=seconds >"${RUN_ROOT}/remote_run_complete.txt"
