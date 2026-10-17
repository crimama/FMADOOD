#!/usr/bin/env bash
set -euo pipefail
RESULTS=/workspace/results_local_compatible/scopead_v18_q95_corruption_comparators_20261013
SUBSPACE="$RESULTS/subspacead"
RAD_OUT="$RESULTS/rad_cache_v2"
SOURCE=/workspace/isolated_runs/table5_fill_20261007/source
PYTHON=/workspace/.local-runtime/python311-cu124-036828932804/bin/python3.11
PYTHONPATH_VALUE=/workspace/.local-runtime/rad_deps_20260930:/workspace/isolated_runs/table5_fill_20261007/source/src
CONFIG=configs/benchmarks/table5_fill_20261007/mvtec_rad_s16_1120.yaml
OBJECTS=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
CONDITIONS=(brightness contrast defocus_blur gaussian_noise)
DATA_ROOT=/home/hunim/Volume/DATA/MVTecAD_corrupted
export PYTHONPATH="$PYTHONPATH_VALUE"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME=/model-cache FMAD_DINOV2_OFFLINE=1
export DINOV3_MODEL_ID=/model-cache/hub/hf_dinov3_vitb16_5931719e
waited=0
while (( waited < 172800 )); do
  all_done=1
  for condition in "${CONDITIONS[@]}"; do
    status="$SUBSPACE/$condition/status.json"
    metrics="$SUBSPACE/$condition/mvtec_subspacead_s16_1120"
    if [[ ! -f "$status" ]] || ! "$PYTHON" - "$status" "$metrics" <<'PY'
import json, pathlib, sys
status=json.load(open(sys.argv[1]))
metric_count=len(list(pathlib.Path(sys.argv[2]).rglob('metrics.json')))
objects=set(status.get('plan', {}).get('objects', []))
completed={item.get('object') for item in status.get('completed', []) if item.get('label') == 'mvtec_subspacead_s16_1120'}
raise SystemExit(0 if status.get('state') == 'complete' and objects == completed and len(objects) == 15 and metric_count == 15 else 1)
PY
    then all_done=0; break; fi
  done
  (( all_done == 1 )) && break
  sleep 60
  waited=$((waited + 60))
done
if (( waited >= 172800 )); then echo 'Timed out; RAD was not started.' >&2; exit 2; fi
mkdir -p "$RAD_OUT"
cd "$SOURCE"
pids=()
for index in "${!CONDITIONS[@]}"; do
  condition=${CONDITIONS[$index]}
  device="cuda:$index"
  output="$RAD_OUT/$condition"
  "$PYTHON" scripts/experiments.py run --configs "$CONFIG" --output-dir "$output" --objects "${OBJECTS[@]}" --timeout 21600 --device "$device" --data-root "$DATA_ROOT/$condition" --split "mvtec_seen_development_v18_corruption_${condition}_rad_cache_seed0_s16" > "$RAD_OUT/${condition}_launch.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit "$rc"
