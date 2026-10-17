#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${REMOTE_WORKDIR:?Set REMOTE_WORKDIR to verified immutable release}"
moment_release="$REMOTE_WORKDIR"
set -a
source configs/remote/dsba3.env
set +a
export REMOTE_WORKDIR="$moment_release"
exec bash scripts/run_experiments.sh --backend remote --container hun_fsad_tta_012 \
  --configs configs/benchmarks/scopead_normal_tail_risk_20261017/ad2_q95_s02.yaml configs/benchmarks/scopead_normal_tail_risk_20261017/ad2_score_free_s02.yaml configs/benchmarks/scopead_normal_tail_risk_20261017/ad2_score_fixed_layer_s02.yaml configs/benchmarks/scopead_normal_tail_risk_20261017/ad2_excess_free_s02.yaml configs/benchmarks/scopead_normal_tail_risk_20261017/ad2_excess_fixed_layer_s02.yaml configs/benchmarks/scopead_normal_tail_risk_20261017/ad2_q95_s16.yaml configs/benchmarks/scopead_normal_tail_risk_20261017/ad2_score_free_s16.yaml configs/benchmarks/scopead_normal_tail_risk_20261017/ad2_score_fixed_layer_s16.yaml configs/benchmarks/scopead_normal_tail_risk_20261017/ad2_excess_free_s16.yaml configs/benchmarks/scopead_normal_tail_risk_20261017/ad2_excess_fixed_layer_s16.yaml \
  --labels ad2_q95_s02 ad2_score_free_s02 ad2_score_fixed_layer_s02 ad2_excess_free_s02 ad2_excess_fixed_layer_s02 ad2_q95_s16 ad2_score_free_s16 ad2_score_fixed_layer_s16 ad2_excess_free_s16 ad2_excess_fixed_layer_s16 \
  --objects can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts \
  --devices GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62 \
  --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
  --output-root /workspace/results_local_compatible/scopead_normal_tail_risk_20261017 \
  --split mvtec_ad2_test_public_seen_development_normal_tail_risk_CQoff_seed0 \
  --timeout 10800 --workers 1
