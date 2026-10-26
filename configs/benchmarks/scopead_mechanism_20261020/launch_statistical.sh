#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
stat_release="$REMOTE_WORKDIR"
source configs/remote/dsba3.env
export REMOTE_WORKDIR="$stat_release"
export REMOTE_PYTHON_HOME=/workspace/.local-runtime/python311-cu124-036828932804
stat_root=/var/tmp/fsad-analysis/scopead_statistical_20261020
gpus=(GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62)
first=(can fabric fruit_jelly rice)
second=(sheet_metal vial wallplugs walnuts)
configs=()
labels=()
for seed in 0 1 2; do
  for shot in 02 16; do
    label="ad2_mechanism_s$shot""_seed$seed"
    labels+=("$label")
    configs+=("configs/benchmarks/scopead_mechanism_20261020/$label.yaml")
  done
done
mkdir -p results/experiment_launch/scopead_statistical_20261020
pids=()
for lane in 0 1 2 3; do
  REMOTE_GPU_DEVICES="${gpus[lane]}" bash scripts/run_experiments.sh \
    --backend remote --container hun_fsad_tta_012 \
    --entrypoint scripts/calibration_statistical_replay.py \
    --configs "${configs[@]}" --labels "${labels[@]}" \
    --objects "${first[lane]}" "${second[lane]}" --devices "${gpus[lane]}" \
    --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
    --output-root "$stat_root/gpu_$lane" \
    --log-dir "results/experiment_launch/scopead_statistical_20261020/gpu_$lane" \
    --split mvtec_ad2_test_public_seen_development_rank_value_statistical_seed012 \
    --after-status "/workspace/results_local_compatible/scopead_mechanism_20261020/lane_$lane/status.json" \
    --timeout 21600 --workers 1 \
    >"results/experiment_launch/scopead_statistical_20261020/gpu_$lane.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
(( rc == 0 )) || exit "$rc"
scripts/run_remote.sh exec -- python3 scripts/calibration_statistical_summary.py \
  --root "$stat_root" --export-root /workspace/results_local_compatible/scopead_statistical_20261020_compact
