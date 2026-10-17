#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
: "${REMOTE_WORKDIR:?Set REMOTE_WORKDIR to verified immutable release}"
moment_release="$REMOTE_WORKDIR"
set -a
source configs/remote/dsba3.env
set +a
export REMOTE_WORKDIR="$moment_release"
export REMOTE_GPU_DEVICES="GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f,GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639"
exec bash scripts/run_experiments.sh --backend remote --container hun_fsad_tta_012   --entrypoint scripts/reference_bank_diagnostic.py   --configs configs/benchmarks/scopead_reference_bank_diagnostic_20261017/ad2_q95_reference_diagnostic_s02.yaml configs/benchmarks/scopead_reference_bank_diagnostic_20261017/ad2_q95_reference_diagnostic_s16.yaml   --labels ad2_q95_reference_diagnostic_s02 ad2_q95_reference_diagnostic_s16   --objects can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts   --devices GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639   --data-root /home/hunim/Volume/DATA/mvtec_ad_2   --output-root /workspace/results_local_compatible/scopead_reference_bank_diagnostic_20261017   --split mvtec_ad2_test_public_seen_development_reference_bank_diagnostic_CQoff_seed0   --timeout 10800 --workers 1
