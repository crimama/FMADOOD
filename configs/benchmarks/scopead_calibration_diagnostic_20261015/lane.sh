#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
G=${1:?lane}
WAIT_PID=${2:-0}
WAIT_START=${3:-0}
ROOT=/workspace/results_local_compatible/scopead_calibration_diagnostic_20261015
CFG=configs/benchmarks/scopead_calibration_diagnostic_20261015
GPUS=(GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62)
OBJECTS=("can fabric" "fruit_jelly rice" "sheet_metal vial" "wallplugs walnuts")
mkdir -p "$ROOT/gpu$G"
exec 8>"$ROOT/gpu$G/launch.lock"
flock -n 8 || exit 2
exec 9>/tmp/fmad-remote-execution.lock
flock -sn 9 || exit 2
export PYTHONPATH=$PWD/src PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME=/model-cache FMAD_DINOV2_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${GPUS[$G]}
printf '%s\n' "$PWD" > "$ROOT/gpu$G/source_release.txt"
cp .fmad-source-sha256 "$ROOT/gpu$G/source_sha256.txt"
printf 'waiting\n' > "$ROOT/gpu$G/queue_state.txt"
printf 'Waiting for PID=%s start=%s on GPU=%s\n' "$WAIT_PID" "$WAIT_START" "$CUDA_VISIBLE_DEVICES"
python3 - "$WAIT_PID" "$WAIT_START" <<'PY'
import sys, time
from pathlib import Path
pid, start = sys.argv[1:]
while True:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        break
    if fields[0] == 'Z' or fields[19] != start:
        break
    time.sleep(30)
PY
printf 'running\n' > "$ROOT/gpu$G/queue_state.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/gpu$G/exit_code.txt"; if [ "$rc" = 0 ]; then echo complete; else echo failed; fi > "$ROOT/gpu$G/queue_state.txt"' EXIT
LABELS=(ad2_none_s02 ad2_tailscale_s02 ad2_none_s16 ad2_tailscale_s16)
CONFIGS=()
for label in "${LABELS[@]}"; do CONFIGS+=("$CFG/$label.yaml"); done
python3 scripts/experiments.py run --configs "${CONFIGS[@]}" --labels "${LABELS[@]}" \
 --objects ${OBJECTS[$G]} --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
 --device cuda:0 --timeout 10800 --workers 1 --output-dir "$ROOT/gpu$G/evaluation" \
 --split mvtec_ad2_test_public_seen_development_calibration_CQoff_seed0
