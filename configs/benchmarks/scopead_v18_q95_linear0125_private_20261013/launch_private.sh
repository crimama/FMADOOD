#!/usr/bin/env bash
# ScopeAD (linear0125, q95) MVTec AD 2 test_private submission, seed 0, 16-shot.
# One lane per existing dsba3 GPU UUID; each lane runs its two objects sequentially.
# Outputs go to the container overlay root (/workspace is nearly full).
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
cd "$D/../../.."
WAIT_PID="${WAIT_PID:-}"
CFG=configs/benchmarks/scopead_v18_q95_linear0125_fp16_20261012/ad2_linear0125_s16_seed0_private.yaml
DATA=/home/hunim/Volume/DATA/mvtec_ad_2
OUT=/var/tmp/scopead_v18_q95_linear0125_private_20261013
mkdir -p "$OUT/logs"
if [ -n "$WAIT_PID" ]; then
  echo "waiting for pid $WAIT_PID ($(date -u +%FT%TZ))"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "pid $WAIT_PID exited ($(date -u +%FT%TZ))"
fi
export PYTHONPATH=$PWD/src PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME=/model-cache FMAD_DINOV2_OFFLINE=1
lane() {
  local uuid="$1"; shift
  export CUDA_VISIBLE_DEVICES="$uuid"
  for obj in "$@"; do
    if [ -f "$OUT/provenance/$obj.json" ]; then
      echo "skip $obj: provenance exists"
      continue
    fi
    python3 scripts/prepare_rsg4_private_submission.py \
      --config "$CFG" --data-root "$DATA" --output-root "$OUT" \
      --object "$obj" --device cuda:0 > "$OUT/logs/$obj.log" 2>&1
    echo "done $obj rc=0 $(date -u +%FT%TZ)"
  done
}
lane GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f can fabric &
lane GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae fruit_jelly rice &
lane GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 sheet_metal vial &
lane GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62 wallplugs walnuts &
wait
python3 - "$OUT" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
tiff = sum(1 for _ in root.glob("anomaly_images/*/*/*.tiff"))
png = sum(1 for _ in root.glob("anomaly_images_thresholded/*/*/*.png"))
print(f"counts tiff={tiff} png={png} expected=4090/4090")
PY
echo "ALL DONE $(date -u +%FT%TZ)"
