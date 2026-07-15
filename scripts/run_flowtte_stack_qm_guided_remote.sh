#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${FLOWTTE_STACK_QM_GUIDED_INTERNAL:-0}" != "1" ]]; then
  PRESET="${REMOTE_PRESET:-${ROOT_DIR}/configs/remote/dsba3.env}"
  if [[ -f "${PRESET}" ]]; then
    # shellcheck disable=SC1090
    source "${PRESET}"
  fi
fi
REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST or provide FLOWTTE_REMOTE_PRESET}"
REMOTE_PORT="${REMOTE_PORT:-2222}"
CONTAINER="${CONTAINER_NAME:-hun_fsad_tta_012}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/fsad_tta}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flowtte_stack_qm_guided_20260713_v1}"
REMOTE_RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${RESULTS_ROOT}/flowtte_gapdecomp_anchor_20260712_v1}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-${ROOT_DIR}/results/remote_runs/dsba3/${RUN_NAME}}"
MODE="${1:-launch}"

ssh_remote() { ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "$@"; }

sync_repo() {
  tar -C "${ROOT_DIR}" -cf - \
    --exclude=.git --exclude=.omx --exclude=.pytest_cache --exclude=.ruff_cache \
    --exclude=.venv --exclude=__pycache__ --exclude=results --exclude=results_remote \
    --exclude=PaperWorks --exclude=configs/remote . | ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" \
    "docker exec -i '${CONTAINER}' bash -lc 'mkdir -p \"${REMOTE_REPO}\" && tar -C \"${REMOTE_REPO}\" -xf -'"
}

launch_remote() {
  sync_repo
  ssh_remote "docker exec '${CONTAINER}' bash -lc '
    set -e
    mkdir -p \"${REMOTE_RUN_ROOT}\"
    if [ -e \"${REMOTE_RUN_ROOT}/remote_run_complete.txt\" ]; then echo completed_run_already_exists; exit 0; fi
    if [ -s \"${REMOTE_RUN_ROOT}/run.pid\" ] && kill -0 \"\$(cat \"${REMOTE_RUN_ROOT}/run.pid\")\" 2>/dev/null; then
      echo run_already_running; exit 0
    fi
    if [ -e \"${REMOTE_RUN_ROOT}/stack_results.json\" ]; then
      echo refusing_to_overwrite_existing_stack_results >&2; exit 1
    fi
    nohup python3 \"${REMOTE_REPO}/scripts/analyze_flowtte_stack_qm_guided.py\" \
      --result-root \"${ANCHOR_ROOT}\" \
      --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
      --output-dir \"${REMOTE_RUN_ROOT}\" --workers 8 \
      >\"${REMOTE_RUN_ROOT}/run.log\" 2>&1 </dev/null &
    echo \$! >\"${REMOTE_RUN_ROOT}/run.pid\"
    echo run_pid=\$!
  '"
}

poll_remote() {
  ssh_remote "docker exec '${CONTAINER}' bash -lc '
    ROOT=\"${REMOTE_RUN_ROOT}\"
    pid=\$(cat \"\$ROOT/run.pid\" 2>/dev/null || true)
    if [ -n \"\$pid\" ] && kill -0 \"\$pid\" 2>/dev/null; then state=running; else state=stopped; fi
    echo pid=\$pid state=\$state
    tail -n 30 \"\$ROOT/run.log\" 2>/dev/null || true
    if [ -s \"\$ROOT/stack_results.json\" ]; then
      python3 - \"\$ROOT\" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
payload = json.loads((root / "stack_results.json").read_text())
print(json.dumps(payload["reference_checks"], sort_keys=True))
if payload["reference_checks_pass"]:
    (root / "remote_run_complete.txt").write_text("status=complete\n")
PY
    fi
    cat \"\$ROOT/remote_run_complete.txt\" 2>/dev/null || true
  '"
}

pull_remote() {
  mkdir -p "${LOCAL_RUN_ROOT}"
  ssh_remote "docker exec '${CONTAINER}' tar -C '${REMOTE_RUN_ROOT}' -cf - ." \
    | tar -C "${LOCAL_RUN_ROOT}" -xf -
}

case "${MODE}" in
  launch) launch_remote ;;
  poll) poll_remote ;;
  pull) pull_remote ;;
  *) echo "usage: $0 [launch|poll|pull]" >&2; exit 2 ;;
esac
