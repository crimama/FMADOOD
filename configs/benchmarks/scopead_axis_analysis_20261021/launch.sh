#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
axis_release="$REMOTE_WORKDIR"
source configs/remote/dsba3.env
export REMOTE_WORKDIR="$axis_release"
export REMOTE_PYTHON_HOME=/workspace/.local-runtime/python311-cu124-036828932804
axis_root=/var/tmp/fsad-analysis/scopead_axis_analysis_20261021
configs=()
labels=()
for seed in 0 1 2; do
  for shot in 02 16; do
    label="ad2_mechanism_s$shot""_seed$seed"
    labels+=("$label")
    configs+=("configs/benchmarks/scopead_mechanism_20261020/$label.yaml")
  done
done
bash scripts/run_experiments.sh --backend remote --container hun_fsad_tta_012 \
  --entrypoint scripts/calibration_axis_replay.py \
  --configs "${configs[@]}" --labels "${labels[@]}" \
  --objects can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts \
  --devices GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62 \
  --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
  --output-root "$axis_root" \
  --log-dir results/experiment_launch/scopead_axis_analysis_20261021 \
  --split mvtec_ad2_test_public_seen_development_axis_mechanism_seed012 \
  --timeout 21600 --workers 1
scripts/run_remote.sh exec -- python3 scripts/calibration_axis_summary.py \
  --root "$axis_root" --export-root /workspace/results_local_compatible/scopead_axis_analysis_20261021_compact
