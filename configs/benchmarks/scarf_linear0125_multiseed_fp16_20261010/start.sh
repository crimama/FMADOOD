#!/usr/bin/env bash
set -euo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
mkdir -p /workspace/results_local_compatible/scarf_linear0125_multiseed_fp16_20261010
for g in 0 1 2 3; do
  nohup bash "$D/shard.sh" $g > /workspace/results_local_compatible/scarf_linear0125_multiseed_fp16_20261010/gpu$g.launch.log 2>&1 < /dev/null &
done
echo started

