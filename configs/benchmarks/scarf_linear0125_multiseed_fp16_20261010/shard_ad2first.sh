#!/usr/bin/env bash
# AD2-first shard runner (replaces shard.sh order). usage: shard_ad2first.sh <gpu 0-3>
# Order: (1) AD2 factorial MAD arm and 1120-only linear arms (A linear MAD 5 + B linear 6 = 11 configs; none and q95 linear arms are in phase 2 because seed-0 AD2 none and q95 results already exist from earlier suites);
#        (2) AD2 multi-seed (seeds 1,2; shots 2,16; linear0125 and deep-only);
#        (3) AD2 factorial control arms (deep-only A/B, cap1024, stage C: 29 configs);
#        (4) MVTec AD and VisA multi-seed (resumes earlier partial MVTec runs).
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
S=configs/benchmarks/scarf_linear0125_multiseed_fp16_20261010
F=configs/benchmarks/scarf_calibration_factorial_seed0_fp16_20261009
OUT=/workspace/results_local_compatible/scarf_linear0125_multiseed_fp16_20261010
OUT2=/workspace/results_local_compatible/scarf_calibration_factorial_seed0_fp16_20261009
mkdir -p "$OUT"
declare -A ROOT=( [ad2]=/home/hunim/Volume/DATA/mvtec_ad_2 [mvtec]=/home/hunim/Volume/DATA/MVTecAD [visa]=/home/hunim/Volume/DATA/VisA_pytorch/1cls )
factorial() { # $1 = order file
  LBL=(); CFG=()
  while read -r l; do LBL+=($l); CFG+=($F/$l.yaml); done < "$1"
  RES=""; [ -f "$OUT2/gpu${G}_$2/status.json" ] && RES=--resume
  python3 scripts/experiments.py run --configs "${CFG[@]}" --labels "${LBL[@]}" --objects "${AD2_OBJ[@]}" --data-root "${ROOT[ad2]}" --output-dir "$OUT2/gpu${G}_$2" --device cuda:0 --timeout 10800 --workers 1 $RES --split "ad2_seen_development_seed0_scarf_calibration_factorial_same_snapshot"
  echo "factorial $1 done rc=$? $(date -u +%FT%TZ)"
}
multiseed() { # $1 = dataset
  ds=$1
  case $ds in ad2) OBJ=("${AD2_OBJ[@]}");; mvtec) OBJ=("${MVTEC_OBJ[@]}");; visa) OBJ=("${VISA_OBJ[@]}");; esac
  LBL=(); CFG=()
  for seed in 1 2; do for shot in 02 16; do for arm in linear0125 deep; do
    l=${ds}_${arm}_s${shot}_seed${seed}; LBL+=($l); CFG+=($S/$l.yaml)
  done; done; done
  RES=""; [ -f "$OUT/gpu${G}_$ds/status.json" ] && RES=--resume
  python3 scripts/experiments.py run --configs "${CFG[@]}" --labels "${LBL[@]}" --objects "${OBJ[@]}" --data-root "${ROOT[$ds]}" --output-dir "$OUT/gpu${G}_$ds" --device cuda:0 --timeout 10800 --workers 1 $RES --split "${ds}_seen_development_seeds12_linear0125_vs_deep_fp16"
  echo "multiseed $ds done rc=$? $(date -u +%FT%TZ)"
}
factorial $S/ad2_factorial_phase1_linear.txt ad2_p1
multiseed ad2
factorial $S/ad2_factorial_phase2_control.txt ad2_ctrl
multiseed mvtec
multiseed visa

