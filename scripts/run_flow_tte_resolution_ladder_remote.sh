#!/usr/bin/env bash
# Phase 5: resolution-only causality ladder, strictly 672 -> 896 -> 1120.
#
# Resolution mechanism: full-frame DINOv3 shorter-edge resize via
# --backbone-resolution. Stage 672 deliberately passes 0 so it follows the
# anchor's exact info.resolution=672 path; 896/1120 pass those explicit values.
# Tiling is disabled at every stage and image-resize-factor remains 1.0 (it is
# only active in the tiled extractor). Token grids are 42², 56², and 70².
# DVT position_mean and the fixed-structure MLP flow are refit independently
# from the fixed support set at each resolution. Query chunks are 512/512/256.
# No TIFFs are written or retained.
#
# Planning estimate (not measured): assume the 672 anchor costs 1.0 GPU-hour
# per object (8.0 GPU-hours/stage), then scale by token count: 672=8.0,
# 896=14.2 (56²/42²), 1120=22.2 GPU-hours (70²/42²). Four-way sharding gives
# approximate wall times 2.0/3.6/5.6 hours if object costs balance. The
# controller records measured summed per-object GPU-hours after every stage.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-launch}"
if [[ "${MODE}" != internal && -f "${ROOT_DIR}/configs/remote/dsba3.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/configs/remote/dsba3.env"
fi
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-2222}"
CONTAINER="${CONTAINER_NAME:-${CONTAINER:-hun_fsad_tta_012}}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/fsad_tta}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
RUN_NAME="${RUN_NAME:-flowtte_resolution_ladder_20260713_v1}"
REMOTE_RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${RESULTS_ROOT}/flowtte_gapdecomp_anchor_20260712_v1}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-${ROOT_DIR}/results/remote_runs/dsba3/${RUN_NAME}}"

ssh_remote() { ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "$@"; }

sync_repo() {
  tar -C "${ROOT_DIR}" -cf - \
    --exclude=.git --exclude=.omx --exclude=.pytest_cache --exclude=.ruff_cache \
    --exclude=.venv --exclude=__pycache__ --exclude=results --exclude=results_remote \
    --exclude=PaperWorks --exclude=configs/remote . | \
    ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" \
      "docker exec -i '${CONTAINER}' bash -lc 'mkdir -p \"${REMOTE_REPO}\" && tar -C \"${REMOTE_REPO}\" -xf -'"
}

poll_remote() {
  ssh_remote "docker exec '${CONTAINER}' bash -lc 'ROOT=\"${REMOTE_RUN_ROOT}\"; \
    printf \"controller_pid=\"; cat \"\$ROOT/controller.pid\" 2>/dev/null || true; \
    cat \"\$ROOT/current_stage.txt\" 2>/dev/null || true; \
    find \"\$ROOT/stages\" -maxdepth 2 -type f \( -name COMPLETE -o -name INVALID \) -print 2>/dev/null; \
    for f in \"\$ROOT\"/stage_*_leaderboard.tsv; do [ -f \"\$f\" ] && { echo ===\"\$f\"; cat \"\$f\"; }; done; \
    cat \"\$ROOT/ladder_summary.json\" 2>/dev/null || true; \
    cat \"\$ROOT/remote_run_complete.txt\" 2>/dev/null || true'"
}

pull_remote() {
  mkdir -p "${LOCAL_RUN_ROOT}"
  ssh_remote "docker exec '${CONTAINER}' tar -C '${REMOTE_RUN_ROOT}' --exclude='*.tiff' --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' -cf - ." | tar -C "${LOCAL_RUN_ROOT}" -xf -
}

launch_remote() {
  [[ -n "${REMOTE_HOST}" ]] || { echo "REMOTE_HOST is required" >&2; exit 2; }
  sync_repo
  ssh_remote "docker exec '${CONTAINER}' bash -lc '
    set -e
    ROOT=\"${REMOTE_RUN_ROOT}\"
    mkdir -p \"\$ROOT\"
    if [ -f \"\$ROOT/remote_run_complete.txt\" ]; then echo run_already_complete; exit 0; fi
    if [ -f \"\$ROOT/stage_672_INVALID.json\" ]; then echo parity_invalid_use_new_RUN_NAME >&2; exit 42; fi
    if [ -s \"\$ROOT/controller.pid\" ] && kill -0 \"\$(cat \"\$ROOT/controller.pid\")\" 2>/dev/null; then echo controller_already_running; exit 0; fi
    if [ -f \"\$ROOT/run_manifest.json\" ]; then
      grep -F '\"run_name\":\"${RUN_NAME}\"' \"\$ROOT/run_manifest.json\" >/dev/null &&
      grep -F '\"resolutions\":[672,896,1120]' \"\$ROOT/run_manifest.json\" >/dev/null || {
        echo manifest_mismatch_use_new_RUN_NAME >&2; exit 3;
      }
    elif find \"\$ROOT\" -mindepth 1 -maxdepth 1 ! -name controller.log -print -quit | grep -q .; then
      echo partial_root_without_manifest_use_new_RUN_NAME >&2; exit 3
    fi
    nohup env FLOWTTE_RESOLUTION_LADDER_INTERNAL=1 RUN_NAME=\"${RUN_NAME}\" RESULTS_ROOT=\"${RESULTS_ROOT}\" DATA_ROOT=\"${DATA_ROOT}\" ANCHOR_ROOT=\"${ANCHOR_ROOT}\" REMOTE_REPO=\"${REMOTE_REPO}\" \
      bash \"${REMOTE_REPO}/scripts/run_flow_tte_resolution_ladder_remote.sh\" internal \
      >\"\$ROOT/controller.log\" 2>&1 </dev/null &
    echo \$! >\"\$ROOT/controller.pid\"
    echo controller_pid=\$!
  '"
}

run_chunk() {
  local gpu="$1" resolution="$2" chunk="$3" objects="$4" backbone_resolution="$5" query_chunk="$6"
  local stage="${REMOTE_RUN_ROOT}/stages/${resolution}" output="${REMOTE_RUN_ROOT}/stages/${resolution}/chunks/${chunk}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 "${REMOTE_REPO}/scripts/run_flow_tte_resolution_ladder.py" \
    --anchor-root "${ANCHOR_ROOT}" --ladder-resolution "${resolution}" \
    --data-root "${DATA_ROOT}" --project-root /workspace --fsad-root "${REMOTE_REPO}" \
    --output-root "${output}" --objects "${objects}" --shots 16 --seed 0 --device cuda \
    --flow-epochs 3 --coupling-layers 2 --hidden-multiplier 1 --flow-lr 2e-4 \
    --flow-clamp 1.9 --tail-weight 0.3 --tail-top-k-ratio 0.05 --lambda-logdet 2e-2 \
    --density-quantile 0.90 --expansion-budget 1.0 --distance-weight 1.0 --density-weight 0.25 \
    --score-mode latent_distance --residual-weight 0.25 --top-percent 0.01 \
    --query-chunk-size "${query_chunk}" --calibration-sample-size 4096 --pro-integration-limit 0.05 \
    --backbone-model dinov3_vith16plus --backbone-resolution "${backbone_resolution}" \
    --feature-layers 7,15,23,31 --tile-patch-size 0 --tile-overlap 0 --image-resize-factor 1.0 \
    --support-brightness-range 0.80,1.20 \
    --support-selection "fixed_json=${REMOTE_REPO}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json" \
    --support-transforms identity --feature-fusion layer_norm_mean --normality-mode fused \
    --context-source none --flow-context-source auto --memory-context-source auto \
    --context-mode none --context-weight 0.0 --context-top-m 1 --flow-condition-mode none \
    --transformer-context-mode none --flow-transform-mode flow \
    --dvt-denoise-mode position_mean --dvt-denoise-alpha 1.0 \
    --score-field-calibration-mode none --score-field-calibration-alpha 1.0 \
    --score-field-position-std-floor 0.25 --score-field-foreground-mode none \
    --score-field-foreground-quantile 0.20 --score-field-background-multiplier 0.50 \
    --score-field-foreground-smooth-kernel 5 --score-field-support-score-quantile 0.90 \
    >"${stage}/logs/${chunk}.log" 2>&1
}

run_stage() {
  local resolution="$1" backbone_resolution="$2" query_chunk="$3"
  local stage="${REMOTE_RUN_ROOT}/stages/${resolution}"
  if [[ -f "${stage}/COMPLETE" && -f "${REMOTE_RUN_ROOT}/stage_${resolution}_leaderboard.tsv" ]]; then return 0; fi
  mkdir -p "${stage}/chunks" "${stage}/logs"
  printf 'resolution=%s\n' "${resolution}" >"${REMOTE_RUN_ROOT}/current_stage.txt"
  local pids=() status=0
  run_chunk 0 "${resolution}" gpu0_can_fabric can,fabric "${backbone_resolution}" "${query_chunk}" & pids+=("$!")
  run_chunk 1 "${resolution}" gpu1_fruit_jelly_rice fruit_jelly,rice "${backbone_resolution}" "${query_chunk}" & pids+=("$!")
  run_chunk 2 "${resolution}" gpu2_vial_wallplugs vial,wallplugs "${backbone_resolution}" "${query_chunk}" & pids+=("$!")
  run_chunk 3 "${resolution}" gpu3_sheet_metal_walnuts sheet_metal,walnuts "${backbone_resolution}" "${query_chunk}" & pids+=("$!")
  ACTIVE_PIDS=("${pids[@]}")
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  ACTIVE_PIDS=()
  [[ "${status}" -eq 0 ]] || return "${status}"
  if find "${stage}" -type f -name '*.tiff' -print -quit | grep -q .; then echo unexpected_tiff >&2; return 3; fi
  python3 "${REMOTE_REPO}/scripts/run_flow_tte_resolution_ladder_controller.py" --run-root "${REMOTE_RUN_ROOT}" --resolution "${resolution}"
}

internal_controller() {
  [[ "${FLOWTTE_RESOLUTION_LADDER_INTERNAL:-0}" == 1 ]] || { echo internal_only >&2; exit 2; }
  export FMAD_DINOV3_OFFLINE=1
  mkdir -p "${REMOTE_RUN_ROOT}/stages"
  echo "$$" >"${REMOTE_RUN_ROOT}/controller.pid"
  cat >"${REMOTE_RUN_ROOT}/run_manifest.json" <<EOF
{"run_name":"${RUN_NAME}","resolutions":[672,896,1120],"stage_order":"resolution-major","scorer":"raw_1nn_only","tile_rule":"disabled_all_stages","support_refit":"DVT_and_flow_per_resolution","seed":0}
EOF
  ACTIVE_PIDS=()
  trap 'for pid in "${ACTIVE_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done' INT TERM EXIT
  run_stage 672 0 512 || exit $?
  run_stage 896 896 512 || exit $?
  run_stage 1120 1120 256 || exit $?
  printf 'run_name=%s\nstages=672,896,1120\nmetrics_only=true\nanomaly_map_tiffs_written=0\n' "${RUN_NAME}" >"${REMOTE_RUN_ROOT}/remote_run_complete.txt"
  trap - INT TERM EXIT
}

case "${MODE}" in
  launch) launch_remote ;;
  poll) [[ -n "${REMOTE_HOST}" ]] || { echo REMOTE_HOST_required >&2; exit 2; }; poll_remote ;;
  pull) [[ -n "${REMOTE_HOST}" ]] || { echo REMOTE_HOST_required >&2; exit 2; }; pull_remote ;;
  internal) internal_controller ;;
  *) echo "usage: $0 [launch|poll|pull]" >&2; exit 2 ;;
esac
