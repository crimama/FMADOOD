#!/usr/bin/env bash
# Table 6 comparator private inference, MVTec AD 2 test_private(+mix), 16-shot seed 0, 1120 input.
# Methods run one after another; each method uses the four dsba3 GPU UUIDs (2 objects per GPU).
# Same threshold rule as ScopeAD: class-wise normal-pixel p95 x 1.421 (no private labels).
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
cd "$D/../../.."
DATA=/home/hunim/Volume/DATA/mvtec_ad_2
ROOT=/var/tmp/comparators_private_1120_20261014
export PYTHONPATH=$PWD/src PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME=/model-cache FMAD_DINOV2_OFFLINE=1
GPUS=(
  GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f
  GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae
  GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639
  GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62
)
OBJ_LANES=("can fabric" "fruit_jelly rice" "sheet_metal vial" "wallplugs walnuts")
METHODS=(
  "patchcore configs/benchmarks/public_1120_20261003/patchcore_s16_seed0.yaml"
  "anomalydino configs/benchmarks/public_1120_20261003/anomalydino_s16_seed0.yaml"
  "superad configs/benchmarks/public_1120_20261003/superad_s16_seed0.yaml"
  "rad configs/benchmarks/public_1120_remaining_20261005/rad_s16_seed0_1120.yaml"
  "subspacead configs/benchmarks/public_1120_remaining_20261005/subspacead_s16_seed0_1120.yaml"
)

lane() {
  local name="$1" cfg="$2" out="$3" uuid="$4"; shift 4
  export CUDA_VISIBLE_DEVICES="$uuid"
  local obj
  for obj in "$@"; do
    if [ -f "$out/provenance/$obj.json" ]; then
      echo "skip $name $obj: provenance exists"
      continue
    fi
    if ! python3 scripts/prepare_rsg4_private_submission.py --config "$cfg" --data-root "$DATA" \
        --output-root "$out" --object "$obj" --device cuda:0 > "$out/logs/$obj.log" 2>&1; then
      echo "FAIL $name $obj $(date -u +%FT%TZ) (see $out/logs/$obj.log)"
      return 1
    fi
    echo "done $name $obj $(date -u +%FT%TZ)"
  done
}

for entry in "${METHODS[@]}"; do
  name="${entry%% *}"; cfg="${entry#* }"
  out="$ROOT/$name"
  mkdir -p "$out/logs"
  echo "start $name $(date -u +%FT%TZ) cfg=$cfg"
  pids=()
  for i in 0 1 2 3; do
    lane "$name" "$cfg" "$out" "${GPUS[$i]}" ${OBJ_LANES[$i]} & pids+=($!)
  done
  rc=0
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  if [ "$rc" -ne 0 ]; then
    echo "STOP $name: a lane failed; later methods not started ($(date -u +%FT%TZ))"
    exit 1
  fi
  python3 - "$out" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
tiff = sum(1 for _ in root.glob("anomaly_images/*/*/*.tiff"))
png = sum(1 for _ in root.glob("anomaly_images_thresholded/*/*/*.png"))
print(f"counts tiff={tiff} png={png} expected=4090/4090")
PY
  echo "finished $name $(date -u +%FT%TZ)"
done
echo "ALL DONE $(date -u +%FT%TZ)"
