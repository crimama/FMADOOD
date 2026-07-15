#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
DATA_ROOT="${DATA_ROOT:?DATA_ROOT is required}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-mvtecad1_visionad_identity_density_s1_2_4_8_16_20260714_v1}"
HOST_TAG="${HOST_TAG:?HOST_TAG is required}"
FEATURE_LAYERS="${FEATURE_LAYERS:-2,3,4,5,6,7,8,9}"
ROOT="${RESULTS_ROOT}/${RUN_NAME}"
ALL="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"
FIRST="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut"
LAST="pill,screw,tile,toothbrush,transistor,wood,zipper"
mkdir -p "${ROOT}/logs"
cd "${FSAD_ROOT}"

run_part() {
  local gpu="$1" shot="$2" part="$3" objects="$4"
  local out="${ROOT}/shot_${shot}_${part}"
  test -f "${out}/run_manifest.json" && return 0
  CUDA_VISIBLE_DEVICES="${gpu}" FMAD_DINOV2_OFFLINE=1 python3 scripts/run_flow_tte_mvtec_ad1.py \
    --data-root "${DATA_ROOT}" --output-root "${out}" \
    --project-root /workspace --fsad-root "${FSAD_ROOT}" \
    --objects "${objects}" --shots "${shot}" --seed 0 --device cuda \
    --backbone-model dinov2_vitb14_reg --preprocess-recipe visionad_official \
    --image-size 448 --crop-size 392 --feature-layers "${FEATURE_LAYERS}" \
    --feature-fusion visionad_mean_l2 --support-selection visionad_seeded_random \
    --support-selection-seed 0 --support-transforms identity \
    --support-brightness-range 1.0,1.0 --flow-epochs 3 --coupling-layers 2 \
    --hidden-multiplier 1 --flow-lr 2e-4 --flow-clamp 1.9 \
    --flow-transform-mode identity --tail-weight 0.3 --tail-top-k-ratio 0.05 \
    --lambda-logdet 1e-3 --density-quantile 0.90 --expansion-budget 1.0 \
    --distance-weight 1.0 --density-weight 0.25 --score-mode latent_distance \
    --dvt-denoise-mode none --normality-mode fused --top-percent 0.01 \
    --query-chunk-size 512 --calibration-sample-size 0 --loo-standardization off \
    --cleanup-maps >"${ROOT}/logs/shot_${shot}_${part}.log" 2>&1
}

pids=()
case "${HOST_TAG}" in
  dsba3)
    run_part 0 1 all "${ALL}" & pids+=("$!")
    run_part 1 2 all "${ALL}" & pids+=("$!")
    run_part 2 4 all "${ALL}" & pids+=("$!")
    run_part 3 8 all "${ALL}" & pids+=("$!")
    ;;
  dsba5)
    run_part 0 16 first "${FIRST}" & pids+=("$!")
    run_part 1 16 last "${LAST}" & pids+=("$!")
    ;;
  *) echo "unknown HOST_TAG=${HOST_TAG}" >&2; exit 2 ;;
esac
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
test "${status}" -eq 0
date -u +%FT%TZ >"${ROOT}/${HOST_TAG}_complete.txt"
