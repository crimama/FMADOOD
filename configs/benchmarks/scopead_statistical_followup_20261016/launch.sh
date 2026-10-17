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
  --configs configs/benchmarks/scopead_statistical_followup_20261016/ad2_q95_s02.yaml configs/benchmarks/scopead_statistical_followup_20261016/ad2_eb_mean_scale_s02.yaml configs/benchmarks/scopead_statistical_followup_20261016/ad2_eb_rms_scale_s02.yaml configs/benchmarks/scopead_statistical_followup_20261016/ad2_weibull_bank_q95_s02.yaml configs/benchmarks/scopead_statistical_followup_20261016/ad2_covariance_q95_s02.yaml configs/benchmarks/scopead_statistical_followup_20261016/ad2_q95_s16.yaml configs/benchmarks/scopead_statistical_followup_20261016/ad2_eb_mean_scale_s16.yaml configs/benchmarks/scopead_statistical_followup_20261016/ad2_eb_rms_scale_s16.yaml configs/benchmarks/scopead_statistical_followup_20261016/ad2_weibull_bank_q95_s16.yaml configs/benchmarks/scopead_statistical_followup_20261016/ad2_covariance_q95_s16.yaml \
  --labels ad2_q95_s02 ad2_eb_mean_scale_s02 ad2_eb_rms_scale_s02 ad2_weibull_bank_q95_s02 ad2_covariance_q95_s02 ad2_q95_s16 ad2_eb_mean_scale_s16 ad2_eb_rms_scale_s16 ad2_weibull_bank_q95_s16 ad2_covariance_q95_s16 \
  --objects can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts \
  --devices GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62 \
  --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
  --output-root /workspace/results_local_compatible/scopead_statistical_followup_20261016 \
  --split mvtec_ad2_test_public_seen_development_statistical_followup_CQoff_seed0 \
  --timeout 10800 --workers 1 \
  --after-status /workspace/results_local_compatible/scopead_moment_normalization_20261016/lane_0/status.json /workspace/results_local_compatible/scopead_moment_normalization_20261016/lane_1/status.json /workspace/results_local_compatible/scopead_moment_normalization_20261016/lane_2/status.json /workspace/results_local_compatible/scopead_moment_normalization_20261016/lane_3/status.json
