#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${REMOTE_WORKDIR:?Set REMOTE_WORKDIR to immutable release}"
agg_release="$REMOTE_WORKDIR"
set -a
source configs/remote/dsba3.env
set +a
export REMOTE_WORKDIR="$agg_release"
gpus=(GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639 GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62)
export REMOTE_GPU_DEVICES="$(IFS=,; echo "${gpus[*]}")"
cfg=configs/benchmarks/scopead_joint_layer_scale_20261017
exec bash scripts/run_experiments.sh --backend remote --container hun_fsad_tta_012 \
  --configs "$cfg/ad2_res_order_linear_deep_mean_s02.yaml" "$cfg/ad2_res_order_linear_deep_power15_s02.yaml" "$cfg/ad2_res_order_linear_deep_power20_s02.yaml" "$cfg/ad2_res_order_linear_deep_order_s02.yaml" "$cfg/ad2_res_powertop3_deep_power15_s02.yaml" "$cfg/ad2_res_powertop3_deep_power20_s02.yaml" \
  --labels ad2_res_order_linear_deep_mean_s02 ad2_res_order_linear_deep_power15_s02 ad2_res_order_linear_deep_power20_s02 ad2_res_order_linear_deep_order_s02 ad2_res_powertop3_deep_power15_s02 ad2_res_powertop3_deep_power20_s02 \
  --objects can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts \
  --devices "${gpus[@]}" --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
  --output-root /workspace/results_local_compatible/scopead_joint_layer_scale_20261017 \
  --split mvtec_ad2_test_public_seen_development_joint_layer_scale_CQoff_seed0 \
  --timeout 10800 --workers 1
