#!/usr/bin/env bash
set -euo pipefail
cd /workspace/.local-sync/current
mkdir -p /results/scarf_component_ablation_local_seed0_fp16_20261009
nohup bash configs/benchmarks/scarf_component_ablation_local_seed0_fp16_20261009/dsba3_run_shard.sh 0 > /results/scarf_component_ablation_local_seed0_fp16_20261009/gpu0.launch.log 2>&1 < /dev/null &
nohup bash configs/benchmarks/scarf_component_ablation_local_seed0_fp16_20261009/dsba3_run_shard.sh 1 > /results/scarf_component_ablation_local_seed0_fp16_20261009/gpu1.launch.log 2>&1 < /dev/null &
nohup bash configs/benchmarks/scarf_component_ablation_local_seed0_fp16_20261009/dsba3_run_shard.sh 2 > /results/scarf_component_ablation_local_seed0_fp16_20261009/gpu2.launch.log 2>&1 < /dev/null &
nohup bash configs/benchmarks/scarf_component_ablation_local_seed0_fp16_20261009/dsba3_run_shard.sh 3 > /results/scarf_component_ablation_local_seed0_fp16_20261009/gpu3.launch.log 2>&1 < /dev/null &
echo started
