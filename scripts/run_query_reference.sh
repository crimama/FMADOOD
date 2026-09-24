#!/usr/bin/env bash
set -euo pipefail

# One bounded development suite, not another queue implementation.
# Hypothesis: real-query shared reference adjustment reduces normal false
# positives without absorbing local defects. QR1 -> QR2 -> QR3 each adds one
# switch. QR0 must reproduce R1. No synthesis, gradient update, maps or weights.
# Locked: seen TESTpublic, seed0, normal16, full8, AUPRO@.05 + pAUROC@.05.
# Joint goal: AUPRO >= QR0 and pAUROC > QR0; no automatic tuning after results.
# Budget: 4 configs x 8 classes, 5 GPU lanes, at most 4h per class/config.
# Any lane failure stops that lane; unrelated lanes finish. Runtime/coverage
# failures are invalid runs, not evidence against the hypothesis.
# FastRef motivates query-conditioned references, but this uses one-step,
# bounded spatially-shared translation, trimmed residuals and block exclusion.
# It is not a FastRef reproduction or an established novelty claim.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
# Reuse this reference-adaptation recipe; sequencing remains in the generic CLI.
# SUITE=reference_consensus runs RC0=QR1, RC1=consensus weights,
# RC2=QR1 norm matching, RC3=reference-cell balancing. Same fixed 32-run budget.
# SUITE=mode_reference runs MC0=QR1, MC1=normal-mode innovations,
# MC2=normal-subspace projection, MC3=orthogonal residual gate. Same budget;
# compare AUPRO >= MC0 and pAUROC > MC0, without automatic tuning.
# SUITE=validated_reference runs VR0=QR1, VR1=MC1, VR2=held-block innovations,
# VR3=nested cross-region cosine validation. Same budget; no source PCA gate.
# SUITE=correspondence_guard keeps MC1 corrections and adds mode then identity
# correspondence costs. Remote GPUs each own a stage, two class workers/GPU.
SUITE="${SUITE:-query_reference}"
WORKERS=1
REMOTE_SHARD_BY=objects
case "$SUITE" in
    query_reference)
        CONFIGS=(configs/methods/query_reference_QR{0,1,2,3}.yaml)
        LABELS=(QR0 QR1 QR2 QR3) ;;
    reference_consensus)
        CONFIGS=(configs/methods/reference_consensus_RC{0,1,2,3}.yaml)
        LABELS=(RC0 RC1 RC2 RC3) ;;
    mode_reference)
        CONFIGS=(configs/methods/mode_reference_MC{0,1,2,3}.yaml)
        LABELS=(MC0 MC1 MC2 MC3) ;;
    validated_reference)
        CONFIGS=(configs/methods/validated_reference_VR{0,1,2,3}.yaml)
        LABELS=(VR0 VR1 VR2 VR3) ;;
    correspondence_guard)
        CONFIGS=(configs/methods/correspondence_guard_CG{0,1,2,3}.yaml)
        LABELS=(CG0 CG1 CG2 CG3)
        WORKERS=2
        REMOTE_SHARD_BY=stages ;;
    *) echo 'Unknown reference suite' >&2; exit 2 ;;
esac
RUN_ID="${RUN_ID:-${SUITE}_20260914}"
[[ "$RUN_ID" == "${SUITE}_"* && "$RUN_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid RUN_ID' >&2; exit 2; }
[[ $# -eq 0 || ( $# -eq 1 && "$1" == --dry-run ) ]] || { echo 'Usage: bash scripts/run_query_reference.sh [--dry-run]' >&2; exit 2; }
set -a
source "${REMOTE_PRESET:-configs/remote/dsba3.env}"
set +a
[[ "$REMOTE_CODE_MODE" == local ]] || { echo 'Requires the configured local snapshot mode.' >&2; exit 2; }
LOCAL_GPU=GPU-0070b668-714f-bcbb-6c46-f0b68c9b7f6b
REMOTE_GPUS=(GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f
             GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae
             GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639
             GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62)
REMOTE_GPU_DEVICES="$(IFS=,; echo "${REMOTE_GPUS[*]}")"
export REMOTE_GPU_DEVICES
LAUNCH=(bash scripts/run_experiments.sh --backend both --container "$CONTAINER_NAME"
        --configs "${CONFIGS[@]}" --labels "${LABELS[@]}"
        --local-device "$LOCAL_GPU" --local-objects can fabric
        --remote-devices "${REMOTE_GPUS[@]}"
        --remote-objects fruit_jelly rice sheet_metal vial wallplugs walnuts
        --data-root /data/mvtec_ad_2 --remote-data-root /data/mvtec_ad_2
        --output-root "/results/$RUN_ID"
        --log-dir "$PROJECT_ROOT/results/local_runs/bcma_docker_20260911/$RUN_ID/launch"
        --split test_public_seen_development --timeout 14400 --workers "$WORKERS"
        --remote-shard-by "$REMOTE_SHARD_BY")
if [[ "${1:-}" == --dry-run ]]; then
    printf '%q ' "${LAUNCH[@]}"; printf '\n'
    exit 0
fi

# Validate both fresh output roots before launching either side. No overwrite,
# container replacement, dependency install, Git commit or push is performed.
docker inspect fsad-tta-local-bcma >/dev/null
if [[ "$(docker inspect -f '{{.State.Running}}' fsad-tta-local-bcma)" != true ]]; then
    docker start fsad-tta-local-bcma >/dev/null
fi
docker exec fsad-tta-local-bcma test ! -e "/results/$RUN_ID/local"
scripts/run_remote.sh exec -- test ! -e "/results/$RUN_ID/remote"
scripts/run_remote.sh update-code
exec "${LAUNCH[@]}"
