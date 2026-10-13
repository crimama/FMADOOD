#!/usr/bin/env bash
set -uo pipefail
G="${1:?gpu}"
cd "$(dirname "$0")/../../.."
case "$G" in
  0) UUID=GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f; OBJ=(can fabric) ;;
  1) UUID=GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae; OBJ=(fruit_jelly rice) ;;
  2) UUID=GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639; OBJ=(sheet_metal vial) ;;
  3) UUID=GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62; OBJ=(wallplugs walnuts) ;;
esac
export PYTHONPATH=$PWD/src PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME=/model-cache FMAD_DINOV2_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$UUID"
S=configs/benchmarks/scarf_calibration_variants_fp16_20261011
OUT=/workspace/results_local_compatible/scarf_calibration_variants_fp16_20261011
mkdir -p "$OUT"
LBL=(); CFG=()
while read -r l; do LBL+=($l); CFG+=($S/$l.yaml); done < $S/order.txt
RES=""; [ -f "$OUT/gpu${G}_ad2/status.json" ] && RES=--resume
python3 scripts/experiments.py run --configs "${CFG[@]}" --labels "${LBL[@]}" --objects "${OBJ[@]}" --data-root /home/hunim/Volume/DATA/mvtec_ad_2 --output-dir "$OUT/gpu${G}_ad2" --device cuda:0 --timeout 10800 --workers 1 $RES --split ad2_seen_development_seed0_calibration_variants_fp16
echo "done rc=$? $(date -u +%FT%TZ)"

