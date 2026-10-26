#!/usr/bin/env bash
# Usage: REMOTE_WORKDIR=<staged release> bash launch.sh [ad2 mvtec visa]
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${REMOTE_WORKDIR:?Set REMOTE_WORKDIR to immutable release}"
exp_release="$REMOTE_WORKDIR"
set -a
source configs/remote/dsba3.env
set +a
export REMOTE_WORKDIR="$exp_release"
export REMOTE_PYTHON_HOME=/workspace/.local-runtime/python311-cu124-036828932804
gpus=(GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62)
export REMOTE_GPU_DEVICES="$(IFS=,; echo "${gpus[*]}")"
cfg=configs/benchmarks/scopead_multiseed_renyi_20261019
run_ds() {
  local ds="$1" data="$2"; shift 2
  local -a objs=("$@") labels=() configs=()
  while read -r l; do labels+=("$l"); configs+=("$cfg/$l.yaml"); done < "$cfg/$ds.txt"
  bash scripts/run_experiments.sh --backend remote --container hun_fsad_tta_012 \
    --configs "${configs[@]}" --labels "${labels[@]}" --objects "${objs[@]}" \
    --devices "${gpus[@]}" --data-root "$data" \
    --output-root "/workspace/results_local_compatible/scopead_multiseed_renyi_20261019/$ds" \
    --split "${ds}_seen_development_scopead_multiseed_renyi" --timeout 10800 --workers 1
}
want=("${@:-ad2 mvtec visa}")
for ds in ${want[@]}; do
  case "$ds" in
    ad2) run_ds ad2 /home/hunim/Volume/DATA/mvtec_ad_2 can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts ;;
    mvtec) run_ds mvtec /home/hunim/Volume/DATA/MVTecAD bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper ;;
    visa) run_ds visa /home/hunim/Volume/DATA/VisA_pytorch/1cls candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum ;;
  esac
done
