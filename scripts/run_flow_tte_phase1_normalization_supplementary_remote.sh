#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-launch}"
REMOTE_PRESET="${REMOTE_PRESET:-${ROOT_DIR}/configs/remote/dsba3.env}"
if [[ "${MODE}" != "internal" && -f "${REMOTE_PRESET}" ]]; then
  # shellcheck disable=SC1090
  source "${REMOTE_PRESET}"
fi
if [[ "${MODE}" != "internal" ]]; then
  : "${REMOTE_HOST:?REMOTE_HOST must be set directly or by configs/remote/dsba3.env}"
fi

REMOTE_PORT="${REMOTE_PORT:-22}"
CONTAINER="${CONTAINER_NAME:-hun_fsad_tta_012}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/fsad_tta}"
ANCHOR_ROOT="${ANCHOR_ROOT:-/workspace/results_remote/flowtte_gapdecomp_anchor_20260712_v1}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/results_remote/flowtte_phase1_normalization_20260712_v1}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-${ROOT_DIR}/results/remote_runs/dsba3/flowtte_phase1_normalization_20260712_v1}"

VARIANT_Q="condition_group_quantile_match_to_regular_q4096"
VARIANT_A="condition_tail_affine_to_regular"
RUN_LOG="supplementary_run.log"
RUN_PID="supplementary_run.pid"
RUN_COMPLETE="supplementary_run_complete.txt"

ssh_remote() {
  ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "$@"
}

sync_repo() {
  tar -C "${ROOT_DIR}" -cf - \
    --exclude=.git --exclude=.omx --exclude=.pytest_cache \
    --exclude=.ruff_cache --exclude=.venv --exclude=__pycache__ \
    --exclude=results --exclude=results_remote --exclude=PaperWorks \
    --exclude=configs/remote \
    . | ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" \
    "docker exec -i '${CONTAINER}' bash -lc 'mkdir -p \"${REMOTE_REPO}\" && tar -C \"${REMOTE_REPO}\" -xf -'"
}

launch_remote() {
  sync_repo
  ssh_remote "docker exec '${CONTAINER}' bash -lc '
    set -e
    test -d \"${ANCHOR_ROOT}\"
    test -d \"${OUTPUT_DIR}\"
    for name in \
      \"${VARIANT_Q}.json\" \
      \"${VARIANT_A}.json\" \
      supplementary_leaderboard.tsv \
      supplementary_metadata.json \
      \"${RUN_LOG}\" \"${RUN_PID}\" \"${RUN_COMPLETE}\"; do
      if [ -e \"${OUTPUT_DIR}/\${name}\" ]; then
        echo \"refusing to overwrite existing supplementary artifact: ${OUTPUT_DIR}/\${name}\" >&2
        exit 1
      fi
    done
    nohup env REMOTE_REPO=\"${REMOTE_REPO}\" ANCHOR_ROOT=\"${ANCHOR_ROOT}\" \
      DATA_ROOT=\"${DATA_ROOT}\" OUTPUT_DIR=\"${OUTPUT_DIR}\" \
      bash \"${REMOTE_REPO}/scripts/run_flow_tte_phase1_normalization_supplementary_remote.sh\" internal \
      >\"${OUTPUT_DIR}/${RUN_LOG}\" 2>&1 </dev/null &
    echo \$! >\"${OUTPUT_DIR}/${RUN_PID}\"
    echo supplementary_pid=\$!
  '"
}

run_internal() {
  set +e
  cd "${REMOTE_REPO}"
  python3 scripts/analyze_flowtte_phase1_normalization.py \
    --result-root "${ANCHOR_ROOT}" \
    --data-root "${DATA_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --workers 8 --supplementary-only
  status=$?
  printf "exit_code=%s\nfinished_at=%s\n" "${status}" "$(date -Iseconds)" \
    >"${OUTPUT_DIR}/${RUN_COMPLETE}"
  exit "${status}"
}

poll_remote() {
  ssh_remote "docker exec '${CONTAINER}' bash -lc '
    printf \"pid=\"; cat \"${OUTPUT_DIR}/${RUN_PID}\" 2>/dev/null || true
    if [ -s \"${OUTPUT_DIR}/${RUN_PID}\" ] && kill -0 \"\$(cat \"${OUTPUT_DIR}/${RUN_PID}\")\" 2>/dev/null; then
      echo status=running
    else
      echo status=not_running
    fi
    tail -n 30 \"${OUTPUT_DIR}/${RUN_LOG}\" 2>/dev/null || true
    cat \"${OUTPUT_DIR}/${RUN_COMPLETE}\" 2>/dev/null || true
  '"
}

pull_remote() {
  mkdir -p "${LOCAL_OUTPUT_DIR}"
  ssh_remote "docker exec '${CONTAINER}' tar -C '${OUTPUT_DIR}' -cf - \
    '${VARIANT_Q}.json' '${VARIANT_A}.json' \
    supplementary_leaderboard.tsv supplementary_metadata.json \
    '${RUN_LOG}' '${RUN_PID}' '${RUN_COMPLETE}'" \
    | tar -C "${LOCAL_OUTPUT_DIR}" -xf -
}

case "${MODE}" in
  launch) launch_remote ;;
  poll) poll_remote ;;
  pull) pull_remote ;;
  internal) run_internal ;;
  *) echo "usage: $0 [launch|poll|pull|internal]" >&2; exit 2 ;;
esac
