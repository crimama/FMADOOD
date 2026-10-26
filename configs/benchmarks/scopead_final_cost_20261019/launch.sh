#!/usr/bin/env bash
# One config per GPU lane (no other job on the GPUs), all 8 AD2 classes per lane. REMOTE_WORKDIR = staged release.
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
cfg=configs/benchmarks/scopead_final_cost_20261019
labels=(); configs=()
while read -r l; do labels+=("$l"); configs+=("$cfg/$l.yaml"); done < "$cfg/cost.txt"
exec bash scripts/run_experiments.sh --backend remote --container hun_fsad_tta_012 \
  --configs "${configs[@]}" --labels "${labels[@]}" \
  --objects can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts \
  --devices "${gpus[@]}" --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
  --output-root /workspace/results_local_compatible/scopead_final_cost_20261019 \
  --split mvtec_ad2_test_public_seen_development_final_cost --shard-by stages --timeout 10800 --workers 1
