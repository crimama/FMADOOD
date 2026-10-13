#!/usr/bin/env bash
# dsba3 shard runner: usage shard.sh <gpu 0-3>. Waits for this GPU's VisA factorial worker, then runs
# (1) linear0125 vs deep-only multi-seed (seeds 1,2; shots 2,16; AD2/MVTec/VisA), (2) AD2 calibration factorial.
set -uo pipefail
G="${1:?gpu index}"
cd "$(dirname "$0")/../../.."
case "$G" in
  0) UUID=GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f; AD2_OBJ=(can fabric); MVTEC_OBJ=(bottle grid pill transistor); VISA_OBJ=(candle fryum pcb2) ;;
  1) UUID=GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae; AD2_OBJ=(fruit_jelly rice); MVTEC_OBJ=(cable hazelnut screw wood); VISA_OBJ=(capsules macaroni1 pcb3) ;;
  2) UUID=GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639; AD2_OBJ=(sheet_metal vial); MVTEC_OBJ=(capsule leather tile zipper); VISA_OBJ=(cashew macaroni2 pcb4) ;;
  3) UUID=GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62; AD2_OBJ=(wallplugs walnuts); MVTEC_OBJ=(carpet metal_nut toothbrush); VISA_OBJ=(chewinggum pcb1 pipe_fryum) ;;
  *) echo bad gpu; exit 2 ;;
esac
export PYTHONPATH=$PWD/src PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME=/model-cache FMAD_DINOV2_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$UUID"
while pgrep -f "scarf_calibration_factorial_seed0_fp16_20261009/gpu${G}_visa" >/dev/null; do sleep 60; done
echo "VisA factorial worker for GPU $G finished at $(date -u +%FT%TZ)"
S=configs/benchmarks/scarf_linear0125_multiseed_fp16_20261010
OUT=/workspace/results_local_compatible/scarf_linear0125_multiseed_fp16_20261010
mkdir -p "$OUT"
declare -A ROOT=( [ad2]=/home/hunim/Volume/DATA/mvtec_ad_2 [mvtec]=/home/hunim/Volume/DATA/MVTecAD [visa]=/home/hunim/Volume/DATA/VisA_pytorch/1cls )
for ds in mvtec visa ad2; do
  case $ds in ad2) OBJ=("${AD2_OBJ[@]}");; mvtec) OBJ=("${MVTEC_OBJ[@]}");; visa) OBJ=("${VISA_OBJ[@]}");; esac
  LBL=(); CFG=()
  for seed in 1 2; do for shot in 02 16; do for arm in linear0125 deep; do
    l=${ds}_${arm}_s${shot}_seed${seed}; LBL+=($l); CFG+=($S/$l.yaml)
  done; done; done
  RES=""; [ -f "$OUT/gpu${G}_$ds/status.json" ] && RES=--resume
  python3 scripts/experiments.py run --configs "${CFG[@]}" --labels "${LBL[@]}" --objects "${OBJ[@]}" --data-root "${ROOT[$ds]}" --output-dir "$OUT/gpu${G}_$ds" --device cuda:0 --timeout 10800 --workers 1 $RES --split "${ds}_seen_development_seeds12_linear0125_vs_deep_fp16"
  echo "multiseed $ds done rc=$? $(date -u +%FT%TZ)"
done
F=configs/benchmarks/scarf_calibration_factorial_seed0_fp16_20261009
OUT2=/workspace/results_local_compatible/scarf_calibration_factorial_seed0_fp16_20261009
LBL=(); CFG=()
while read -r l; do LBL+=($l); CFG+=($F/$l.yaml); done < $S/ad2_factorial_order.txt
RES=""; [ -f "$OUT2/gpu${G}_ad2/status.json" ] && RES=--resume
python3 scripts/experiments.py run --configs "${CFG[@]}" --labels "${LBL[@]}" --objects "${AD2_OBJ[@]}" --data-root "${ROOT[ad2]}" --output-dir "$OUT2/gpu${G}_ad2" --device cuda:0 --timeout 10800 --workers 1 $RES --split "ad2_seen_development_seed0_scarf_calibration_factorial_same_snapshot"
echo "AD2 factorial done rc=$? $(date -u +%FT%TZ)"

