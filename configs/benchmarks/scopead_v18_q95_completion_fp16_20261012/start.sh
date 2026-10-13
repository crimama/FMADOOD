#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
O=/workspace/results_local_compatible/scopead_v18_q95_completion_fp16_20261012
mkdir -p "$O"
for g in 0 1 2 3; do nohup bash "$D/shard.sh" "$g" > "$O/gpu${g}.launch.log" 2>&1 < /dev/null & echo "GPU $g pid=$!"; done
