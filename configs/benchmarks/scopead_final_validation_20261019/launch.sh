#!/usr/bin/env bash
# Usage: REMOTE_WORKDIR=<staged release> bash launch.sh <phase>...
# phases: calib_seeds lambda_sweep q_sweep corruption; env GPU_IDX="1 2 3" SHARD_BY=stages|objects
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${REMOTE_WORKDIR:?Set REMOTE_WORKDIR to immutable release}"
exp_release="$REMOTE_WORKDIR"
set -a
source configs/remote/dsba3.env
set +a
export REMOTE_WORKDIR="$exp_release"
export REMOTE_PYTHON_HOME=/workspace/.local-runtime/python311-cu124-036828932804
all_gpus=(GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62)
gpus=()
for i in ${GPU_IDX:-0 1 2 3}; do gpus+=("${all_gpus[$i]}"); done
export REMOTE_GPU_DEVICES="$(IFS=,; echo "${gpus[*]}")"
shard_by="${SHARD_BY:-objects}"
cfg=configs/benchmarks/scopead_final_validation_20261019
out=/workspace/results_local_compatible/scopead_final_validation_20261019
ad2_objs=(can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts)
mvt_objs=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)
run_phase() {
  local phase="$1" list="$2" data="$3" split="$4"; shift 4
  local -a objs=("$@") labels=() configs=()
  while read -r l; do labels+=("$l"); configs+=("$cfg/$l.yaml"); done < "$cfg/$list.txt"
  bash scripts/run_experiments.sh --backend remote --container hun_fsad_tta_012 \
    --configs "${configs[@]}" --labels "${labels[@]}" --objects "${objs[@]}" \
    --devices "${gpus[@]}" --data-root "$data" --output-root "$out/$phase" \
    --split "$split" --shard-by "$shard_by" --timeout 10800 --workers 1
}
for phase in "$@"; do
  case "$phase" in
    calib_seeds|lambda_sweep|q_sweep)
      run_phase "$phase" "$phase" /home/hunim/Volume/DATA/mvtec_ad_2 "mvtec_ad2_test_public_seen_development_final_$phase" "${ad2_objs[@]}" ;;
    corruption)
      for c in clean brightness contrast defocus_blur gaussian_noise; do
        root=/home/hunim/Volume/DATA/MVTecAD_corrupted/$c
        [ "$c" = clean ] && root=/home/hunim/Volume/DATA/MVTecAD
        shard_by=objects run_phase "corruption_$c" corruption_clean "$root" "mvtec_seen_development_final_corruption_$c" "${mvt_objs[@]}"
      done ;;
  esac
done
