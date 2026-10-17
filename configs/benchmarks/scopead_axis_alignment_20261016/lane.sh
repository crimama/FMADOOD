#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
G=${1:?lane}
[[ $G =~ ^[0-3]$ ]]
ROOT=/workspace/results_local_compatible/scopead_axis_alignment_20261016
CFG=configs/benchmarks/scopead_axis_alignment_20261016
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
printf 'running\n' > "$ROOT/gpu$G/queue_state.txt"
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/gpu$G/exit_code.txt"; if [ "$rc" = 0 ]; then echo complete; else echo failed; fi > "$ROOT/gpu$G/queue_state.txt"' EXIT
ARMS=(axis_layer axis_resolution axis_layer_resolution)
LABELS=()
CONFIGS=()
for shot in 02 16; do
  for arm in "${ARMS[@]}"; do
    label=ad2_${arm}_s${shot}
    LABELS+=("$label")
    CONFIGS+=("$CFG/$label.yaml")
  done
done
python3 scripts/experiments.py run --configs "${CONFIGS[@]}" --labels "${LABELS[@]}" \
 --objects ${OBJECTS[$G]} --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
 --device cuda:0 --timeout 10800 --workers 1 --output-dir "$ROOT/gpu$G/evaluation" \
 --split mvtec_ad2_test_public_seen_development_axis_alignment_CQoff_seed0
