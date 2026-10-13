#!/usr/bin/env bash
set -euo pipefail
G="${1:?gpu}"
cd "$(dirname "$0")/../../.."
case "$G" in
 0) UUID=GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f; AD2_OBJ=(can fabric); MVTEC_OBJ=(bottle grid pill transistor); VISA_OBJ=(candle fryum pcb2) ;;
 1) UUID=GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae; AD2_OBJ=(fruit_jelly rice); MVTEC_OBJ=(cable hazelnut screw wood); VISA_OBJ=(capsules macaroni1 pcb3) ;;
 2) UUID=GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639; AD2_OBJ=(sheet_metal vial); MVTEC_OBJ=(capsule leather tile zipper); VISA_OBJ=(cashew macaroni2 pcb4) ;;
 3) UUID=GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62; AD2_OBJ=(wallplugs walnuts); MVTEC_OBJ=(carpet metal_nut toothbrush); VISA_OBJ=(chewinggum pcb1 pipe_fryum) ;;
 *) exit 2 ;;
esac
export PYTHONPATH=$PWD/src PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_HOME=/model-cache FMAD_DINOV2_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$UUID"
S=configs/benchmarks/scopead_v18_q95_linear0125_fp16_20261012
OUT=/workspace/results_local_compatible/scopead_v18_q95_linear0125_fp16_20261012
mkdir -p "$OUT"
run_phase() {
 local phase="$1" ds="$2" data="$3" target="$OUT/gpu${G}_$1"
 local -a OBJ LBL CFG RES
 case "$ds" in ad2) OBJ=("${AD2_OBJ[@]}");; mvtec) OBJ=("${MVTEC_OBJ[@]}");; visa) OBJ=("${VISA_OBJ[@]}");; esac
 LBL=(); CFG=(); RES=()
 while read -r l; do LBL+=("$l"); CFG+=("$S/$l.yaml"); done < "$S/$phase.txt"
 if [ -f "$target/status.json" ]; then
  state=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["state"])' "$target/status.json")
  [ "$state" = complete ] && return 0
  [ "$state" = invalid_run ] || { echo "Unsafe resume state: $state"; return 2; }
  RES=(--resume)
 fi
 python3 scripts/experiments.py run --configs "${CFG[@]}" --labels "${LBL[@]}" --objects "${OBJ[@]}" --data-root "$data" --output-dir "$target" --device cuda:0 --timeout 10800 --workers 1 "${RES[@]}" --split "${ds}_seen_development_v18_q95_${phase}"
}
run_phase capacity_cost_ad2 ad2 /home/hunim/Volume/DATA/mvtec_ad_2
run_phase multiseed_ad2 ad2 /home/hunim/Volume/DATA/mvtec_ad_2
run_phase strata_ad2 ad2 /home/hunim/Volume/DATA/mvtec_ad_2
run_phase multiseed_mvtec mvtec /home/hunim/Volume/DATA/MVTecAD
run_phase multiseed_visa visa /home/hunim/Volume/DATA/VisA_pytorch/1cls
run_phase corruption_clean mvtec /home/hunim/Volume/DATA/MVTecAD
run_phase corruption_brightness mvtec /home/hunim/Volume/DATA/MVTecAD_corrupted/brightness
run_phase corruption_contrast mvtec /home/hunim/Volume/DATA/MVTecAD_corrupted/contrast
run_phase corruption_defocus_blur mvtec /home/hunim/Volume/DATA/MVTecAD_corrupted/defocus_blur
run_phase corruption_gaussian_noise mvtec /home/hunim/Volume/DATA/MVTecAD_corrupted/gaussian_noise
for obj in "${AD2_OBJ[@]}"; do
 while read -r label; do
  profile="$OUT/components/gpu${G}/$obj/$label/metrics.json"
  [ -f "$profile" ] && continue
  python3 scripts/profile_rsg4_components_local.py --config "$S/$label.yaml" --data-root /home/hunim/Volume/DATA/mvtec_ad_2 --object "$obj" --queries 12 --output "$profile"
 done < "$S/capacity_cost_ad2.txt"
done
echo "DONE rc=0 $(date -u +%FT%TZ)"
