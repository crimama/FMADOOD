#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
RUN_ROOT="${2:-/workspace/results_remote/darc_v11_g2_ad1synthetic15_p16r_20260711_g4_v1}"
DATA_ROOT="${DARC_AD1_ROOT:-/home/hunim/Volume/DATA/MVTecAD}"
SCRIPT="/workspace/scripts/run_flow_tte_darc_gate2.py"
GATE1_DECISION="/workspace/results_remote/darc_v11_g1_ad1synthetic15_p16r_20260710_v1/gate_decision.json"

if [[ "${MODE}" != "full" && "${MODE}" != "smoke" ]]; then
  echo "usage: $0 [full|smoke] [run-root]" >&2
  exit 2
fi
if [[ -z "${FMAD_DARC_GATE2_PROFILE+x}" ]]; then
  if [[ "${MODE}" == "smoke" ]]; then
    export FMAD_DARC_GATE2_PROFILE=1
  else
    export FMAD_DARC_GATE2_PROFILE=0
  fi
fi
if [[ ! -f "${SCRIPT}" || ! -d "${DATA_ROOT}" || ! -f "${GATE1_DECISION}" ]]; then
  echo "missing runner, AD1 root, or Gate1 decision" >&2
  exit 2
fi

python3 - "${GATE1_DECISION}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
decision = json.loads(path.read_text(encoding="utf-8"))
if decision.get("passed") is not True or decision.get("source_count") != 720:
    raise SystemExit(f"Gate1 prerequisite did not pass its exact 720-source population: {path}")
PY

mkdir -p "${RUN_ROOT}/logs"
exec 9>"${RUN_ROOT}/controller.lock"
if ! flock -n 9; then
  echo "DARC Gate2 controller already owns ${RUN_ROOT}" >&2
  exit 3
fi

{
  date --iso-8601=seconds
  hostname
  pwd
  df -h /workspace
  nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv,noheader
} >"${RUN_ROOT}/preflight.txt"

run_object() {
  local gpu="$1"
  local object_name="$2"
  exec env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    python3 "${SCRIPT}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${RUN_ROOT}" \
      --object "${object_name}" \
      --device cuda:0 \
      --seeds 0,1,2 \
      --resume \
    >"${RUN_ROOT}/logs/gpu${gpu}_${object_name}.log" 2>&1
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

pids=()
labels=()
launch_object() {
  local gpu="$1"
  local object_name="$2"
  run_object "${gpu}" "${object_name}" &
  pids+=("$!")
  labels+=("gpu${gpu}:${object_name}")
}

trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
launch_object 0 bottle
launch_object 0 carpet
launch_object 0 leather
launch_object 0 screw
launch_object 1 transistor
launch_object 1 cable
launch_object 1 grid
launch_object 1 metal_nut
launch_object 2 tile
launch_object 2 wood
launch_object 2 capsule
launch_object 2 hazelnut
launch_object 3 pill
launch_object 3 toothbrush
launch_object 3 zipper

failures=()
set +e
for index in "${!pids[@]}"; do
  wait "${pids[index]}"
  status=$?
  if (( status != 0 )); then
    failures+=("${labels[index]}=${status}")
  fi
done
set -e
if (( ${#failures[@]} != 0 )); then
  echo "object failures: ${failures[*]}" >&2
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
if len(hashes) != 1 or None in hashes:
    raise SystemExit(f"method hash mismatch: {hashes}")
PY

python3 "${SCRIPT}" --output-root "${RUN_ROOT}" --aggregate \
  >"${RUN_ROOT}/logs/aggregate.log" 2>&1
date --iso-8601=seconds >"${RUN_ROOT}/remote_run_complete.txt"
