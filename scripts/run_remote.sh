#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASKPASS_SCRIPT="${ROOT_DIR}/scripts/ssh_askpass_env.sh"

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
JUMP_HOST="${JUMP_HOST:-}"
JUMP_PORT="${JUMP_PORT:-22}"
REMOTE_REPO_DIR_EXPLICIT_EMPTY=0
if [[ "${REMOTE_REPO_DIR+x}" == x && -z "${REMOTE_REPO_DIR}" ]]; then
  REMOTE_REPO_DIR_EXPLICIT_EMPTY=1
fi
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/tmp/fmad-ood-remote}"
REMOTE_RESULTS_DIR="${REMOTE_RESULTS_DIR:-${REMOTE_REPO_DIR}/results_remote}"
REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data}"
REMOTE_CACHE_ROOT="${REMOTE_CACHE_ROOT:-${REMOTE_REPO_DIR}/cache}"
LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-}"
DATA_SYNC_PATHS="${DATA_SYNC_PATHS:-}"
DOCKERFILE="${DOCKERFILE:-docker/Dockerfile.superad}"
IMAGE_NAME="${IMAGE_NAME:-hun_superad}"
CONTAINER_NAME="${CONTAINER_NAME:-hun_superad_run}"
LOCAL_RESULTS_DIR="${LOCAL_RESULTS_DIR:-${ROOT_DIR}/results/remote_runs}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/workspace}"
FORCE_BUILD="${FORCE_BUILD:-0}"
SYNC_DELETE="${SYNC_DELETE:-0}"
REMOTE_GPU_DEVICES="${REMOTE_GPU_DEVICES:-}"
REMOTE_CODE_MODE="${REMOTE_CODE_MODE:-git}"
REMOTE_LOCAL_ROOT="${REMOTE_LOCAL_ROOT:-/workspace/.local-sync}"
REMOTE_PYTHON_HOME="${REMOTE_PYTHON_HOME:-}"
REMOTE_TORCH_HOME="${REMOTE_TORCH_HOME:-}"
RSYNC_EXCLUDES=(
  "--exclude=.git"
  "--exclude=.claude"
  "--exclude=.codegraph"
  "--exclude=.omx"
  "--exclude=.ruff_cache"
  "--exclude=.venv"
  "--exclude=results"
  "--exclude=results_remote"
  "--exclude=__pycache__"
  "--exclude=.pytest_cache"
  "--exclude=*.pdf"
  "--exclude=*.pth"
  "--exclude=._*"
  "--exclude=MEETING_*"
  "--include=**/*.env.example"
  "--exclude=configs/remote/*.env"
  "--exclude=**/.env"
  "--exclude=**/*.env"
  "--exclude=**/id_rsa"
  "--exclude=**/id_rsa.*"
  "--exclude=**/id_ed25519"
  "--exclude=**/id_ed25519.*"
  "--exclude=**/*.pem"
  "--exclude=**/*.key"
)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_remote.sh <mode> [-- command args...]
  scripts/run_remote.sh pull [--metrics-only]

Modes (infrastructure):
  sync      - rsync project to remote server
  sync-data - rsync datasets to remote server
  build     - build Docker image on remote server

Modes (persistent container):
  start     - reuse running container or start the SAME stopped container
  update-code - git fast-forward, or local snapshot when REMOTE_CODE_MODE=local
               add --stage-only to install a local snapshot without activating it
  exec      - run a command inside the running container
  stop      - explicitly stop the container; NEVER remove it
  logs      - show experiment logs retained in the container
  shell     - open interactive shell in the container

Modes (convenience):
  run       - start (if needed) + exec command
  status    - show container & results status
  pull      - retrieve metrics/config/manifest/summary; --metrics-only omits provenance
  setup     - disabled: provisioning is a separate, explicitly authorized task

Required env:
  REMOTE_HOST=user@server

Example (existing container; local changes must already be committed/pushed):
  source configs/remote/147.47.134.98-via-147.47.39.144.env
  scripts/run_remote.sh start
  scripts/run_remote.sh update-code
  export REMOTE_GPU_DEVICES=GPU-... # UUID of the assigned GPU exposed in container

Example (run experiment — container stays alive):
  scripts/run_remote.sh run -- python3 run.py --device cuda:0 \
    --config configs/default.yaml \
    --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
    --output-dir /workspace/results_remote/exp01

Example (multiple experiments, same container):
  scripts/run_remote.sh exec -- python3 run.py --objects can ...
  scripts/run_remote.sh exec -- python3 run.py --objects fabric ...
  scripts/run_remote.sh pull
EOF
}

require_remote_host() {
  if [[ -z "${REMOTE_HOST}" ]]; then
    echo "REMOTE_HOST is required." >&2
    usage >&2
    exit 1
  fi
}

ssh_base_args() {
  if [[ -n "${JUMP_HOST}" ]]; then
    printf '%s\n' -J "${JUMP_HOST}:${JUMP_PORT}" -p "${REMOTE_PORT}"
  else
    printf '%s\n' -p "${REMOTE_PORT}"
  fi
}

ssh_askpass_env_args() {
  printf '%s\n' env "SSH_PASSWORD=${SSH_PASSWORD}" "DISPLAY=${DISPLAY:-dummy}" "SSH_ASKPASS=${ASKPASS_SCRIPT}" SSH_ASKPASS_REQUIRE=force
}

ssh_cmd() {
  local args=()
  while IFS= read -r line; do
    args+=("${line}")
  done < <(ssh_base_args)
  if [[ -n "${SSH_PASSWORD:-}" ]]; then
    local askpass_env=()
    while IFS= read -r line; do
      askpass_env+=("${line}")
    done < <(ssh_askpass_env_args)
    setsid -w "${askpass_env[@]}" ssh "${args[@]}" "$@" </dev/null
  else
    ssh "${args[@]}" "$@"
  fi
}

rsync_remote_target() {
  printf '[%s]:%s' "${REMOTE_HOST}" "$1"
}

rsync_ssh_cmd() {
  printf 'ssh'
  if [[ -n "${JUMP_HOST}" ]]; then
    printf ' -J %q:%q -p %q' "${JUMP_HOST}" "${JUMP_PORT}" "${REMOTE_PORT}"
  else
    printf ' -p %q' "${REMOTE_PORT}"
  fi
}

run_rsync() {
  if [[ -n "${SSH_PASSWORD:-}" ]]; then
    SSH_PASSWORD="${SSH_PASSWORD}" DISPLAY="${DISPLAY:-dummy}" SSH_ASKPASS="${ASKPASS_SCRIPT}" SSH_ASKPASS_REQUIRE=force setsid -w rsync "$@"
  else
    rsync "$@"
  fi
}

validate_remote_repo_dir_chars() {
  local path="$1"
  case "${path}" in
    *[!A-Za-z0-9_./-]*)
      echo "REMOTE_REPO_DIR contains unsafe shell characters." >&2
      exit 1
      ;;
  esac
}

validate_remote_repo_dir() {
  local path="$1"
  validate_remote_repo_dir_chars "${path}"
  if [[ -z "${path}" || "${path}" != /* ]]; then
    echo "REMOTE_REPO_DIR must be a normalized absolute non-root path." >&2
    exit 1
  fi
  case "${path}" in
    /|/*//*|*/./*|*/../*|*/.|*/..|*/)
      echo "REMOTE_REPO_DIR must be a normalized absolute non-root path." >&2
      exit 1
      ;;
  esac
}

detect_local_data_root() {
  if [[ -n "${LOCAL_DATA_ROOT}" ]]; then
    printf '%s\n' "${LOCAL_DATA_ROOT}"
    return 0
  fi
  if [[ -d /Volume/DATA ]]; then
    printf '%s\n' /Volume/DATA
    return 0
  fi
  if [[ -d /home/hun/Volume/DATA ]]; then
    printf '%s\n' /home/hun/Volume/DATA
    return 0
  fi
  return 1
}

sync_data() {
  require_remote_host
  local resolved_local_data_root
  resolved_local_data_root="$(detect_local_data_root)" || {
    echo "Could not resolve LOCAL_DATA_ROOT. Set LOCAL_DATA_ROOT explicitly." >&2
    exit 1
  }
  if [[ -z "${DATA_SYNC_PATHS}" ]]; then
    echo "DATA_SYNC_PATHS is required for sync-data/setup. Example: DATA_SYNC_PATHS='mvtec_ad_2'" >&2
    exit 1
  fi
  ssh_cmd "${REMOTE_HOST}" "mkdir -p '${REMOTE_DATA_ROOT}'"
  local path
  for path in ${DATA_SYNC_PATHS}; do
    if [[ ! -e "${resolved_local_data_root}/${path}" ]]; then
      echo "Local data path missing: ${resolved_local_data_root}/${path}" >&2
      exit 1
    fi
    echo "Syncing dataset: ${path} ..."
    run_rsync -az -e "$(rsync_ssh_cmd)" "${resolved_local_data_root}/${path}" "$(rsync_remote_target "${REMOTE_DATA_ROOT}/")"
  done
  echo "Data sync complete."
}

sync_repo() {
  require_remote_host
  mkdir -p "${LOCAL_RESULTS_DIR}"
  local delete_flag=()
  if [[ "${SYNC_DELETE}" == "1" ]]; then
    if [[ "${REMOTE_REPO_DIR_EXPLICIT_EMPTY}" == "1" ]]; then
      echo "REMOTE_REPO_DIR must not be explicitly empty when SYNC_DELETE=1." >&2
      exit 1
    fi
    validate_remote_repo_dir "${REMOTE_REPO_DIR}"
    delete_flag=(--delete)
  fi
  ssh_cmd "${REMOTE_HOST}" "mkdir -p '${REMOTE_REPO_DIR}' '${REMOTE_RESULTS_DIR}' '${REMOTE_CACHE_ROOT}/torch/hub' '${REMOTE_CACHE_ROOT}/torch/checkpoints' '${REMOTE_CACHE_ROOT}/huggingface' '${REMOTE_CACHE_ROOT}/pip'"
  echo "Syncing project to remote..."
  run_rsync -az -e "$(rsync_ssh_cmd)" "${delete_flag[@]}" \
    "${RSYNC_EXCLUDES[@]}" \
    "${ROOT_DIR}/" "$(rsync_remote_target "${REMOTE_REPO_DIR}/")"
  echo "Project sync complete."
}

build_image() {
  require_remote_host
  local build_cmd="cd '${REMOTE_REPO_DIR}' && docker build -t '${IMAGE_NAME}' -f '${DOCKERFILE}' ."
  if [[ "${FORCE_BUILD}" == "1" ]]; then
    echo "Force building Docker image..."
    ssh_cmd "${REMOTE_HOST}" "${build_cmd}"
  else
    echo "Building Docker image (skipping if exists)..."
    ssh_cmd "${REMOTE_HOST}" "docker image inspect '${IMAGE_NAME}' >/dev/null 2>&1 || (${build_cmd})"
  fi
  echo "Image ready."
}

# ── Persistent container management ──────────────────────────────────────

is_container_running() {
  ssh_cmd "${REMOTE_HOST}" "docker ps -q --filter name='^${CONTAINER_NAME}$'" 2>/dev/null | grep -q .
}

is_container_exists() {
  ssh_cmd "${REMOTE_HOST}" "docker ps -aq --filter name='^${CONTAINER_NAME}$'" 2>/dev/null | grep -q .
}

start_container() {
  require_remote_host
  if is_container_running; then
    echo "Container '${CONTAINER_NAME}' already running."
    return 0
  fi

  if ! is_container_exists; then
    echo "Existing container '${CONTAINER_NAME}' not found. No container was created." >&2
    return 1
  fi
  echo "Starting existing container '${CONTAINER_NAME}'..."
  ssh_cmd "${REMOTE_HOST}" "docker start '${CONTAINER_NAME}'" >/dev/null
  echo "Container started."
}

stop_container() {
  require_remote_host
  if is_container_running; then
    echo "Stopping container '${CONTAINER_NAME}'..."
    ssh_cmd "${REMOTE_HOST}" "docker stop '${CONTAINER_NAME}'" >/dev/null
  fi
  echo "Container retained (not removed)."
}

container_bash() {
  local command="$1" quoted workdir="${2:-${REMOTE_WORKDIR}}" runtime=""
  if [[ -n "${REMOTE_PYTHON_HOME}" ]]; then
    runtime="export PYTHONHOME='${REMOTE_PYTHON_HOME}'
export PATH='${REMOTE_PYTHON_HOME}/bin':\$PATH
export PYTHONPATH='${REMOTE_WORKDIR}/src'
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"
  else
    runtime="export PYTHONPATH='${REMOTE_WORKDIR}/src'
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
"
  fi
  if [[ -n "${REMOTE_TORCH_HOME}" ]]; then
    runtime+="export TORCH_HOME='${REMOTE_TORCH_HOME}' FMAD_DINOV2_OFFLINE=1
"
  fi
  command="${runtime}${command}"
  # Quote a complete script for the remote login shell; argv inside is Bash-quoted.
  quoted="${command//\'/\'\\\'\'}"
  ssh_cmd "${REMOTE_HOST}" "docker exec -w '${workdir}' -e CUDA_VISIBLE_DEVICES='${REMOTE_GPU_DEVICES}' '${CONTAINER_NAME}' bash -lc '${quoted}'"
}

update_local_code() (
  local stage_only="${1:-0}"
  # A new local-only release cannot retain files from the legacy checkout.
  # Uploads do not touch active code; activation shares the experiment lock.
  validate_remote_repo_dir "${REMOTE_REPO_DIR}"
  validate_remote_repo_dir "${REMOTE_LOCAL_ROOT}"
  [[ "${REMOTE_WORKDIR}" == "${REMOTE_LOCAL_ROOT}/current" ]] || {
    echo "Local mode requires REMOTE_WORKDIR=REMOTE_LOCAL_ROOT/current." >&2; exit 1;
  }
  local snapshot_dir snapshot digest archive remote_archive sources
  snapshot_dir="$(mktemp -d)"
  snapshot="${snapshot_dir}/snapshot.tar"
  trap 'rm -f -- "$snapshot"; rmdir -- "$snapshot_dir"' EXIT
  sources=(run.py pyproject.toml src configs scripts tests docker)
  tar --sort=name --mtime=@0 --owner=0 --group=0 --numeric-owner \
    --exclude=configs/remote --exclude=__pycache__ --exclude=.pytest_cache \
    --exclude=.git --exclude='*.pyc' --exclude='*.env' --exclude='*.pem' \
    --exclude='*.key' --exclude='id_rsa*' --exclude='id_ed25519*' \
    --exclude='*.pth' --exclude='*.pt' \
    -C "${ROOT_DIR}" -cf "${snapshot}" "${sources[@]}"
  digest="$(sha256sum "${snapshot}")"
  digest="${digest%% *}"
  start_container
  # Query from /workspace: current does not exist on first setup.
  local current
  current="$(container_bash "readlink '${REMOTE_LOCAL_ROOT}/current' || true" /workspace)"
  if [[ "${current}" == "${REMOTE_LOCAL_ROOT}/releases/${digest}" ]]; then
    echo "Local code unchanged (${digest}); no transfer or activation."
    exit 0
  fi
  archive="${REMOTE_REPO_DIR}/.local-sync/incoming/${digest}.tar"
  remote_archive="${REMOTE_LOCAL_ROOT}/incoming/${digest}.tar"
  ssh_cmd "${REMOTE_HOST}" "mkdir -p '${REMOTE_REPO_DIR}/.local-sync/incoming'"
  run_rsync -az -e "$(rsync_ssh_cmd)" "${snapshot}" "$(rsync_remote_target "${archive}")"
  if [[ "${stage_only}" == "1" ]]; then
    container_bash "set -euo pipefail
exec 9>/tmp/fmad-remote-execution.lock
flock -sn 9 || { echo 'Exclusive execution/update in progress; release not staged.' >&2; exit 1; }
exec 8>/tmp/fmad-remote-release-stage.lock
flock -n 8 || { echo 'Another release staging operation is in progress.' >&2; exit 1; }
test \"\$(sha256sum '${remote_archive}' | cut -d ' ' -f 1)\" = '${digest}'
current_before=\$(readlink '${REMOTE_LOCAL_ROOT}/current' || true)
mkdir -p '${REMOTE_LOCAL_ROOT}/releases'
release='${REMOTE_LOCAL_ROOT}/releases/${digest}'
if [[ -d \"\$release\" ]]; then
  test \"\$(cat \"\$release/.fmad-source-sha256\")\" = '${digest}'
  current_after=\$(readlink '${REMOTE_LOCAL_ROOT}/current' || true)
  test \"\$current_before\" = \"\$current_after\"
  printf 'Current release before staging: %s\\nCurrent release after staging: %s\\n' \"\$current_before\" \"\$current_after\"
  echo 'Release already staged: ${digest}'
  exit 0
fi
test ! -e \"\$release\"
staging=\$(mktemp -d '${REMOTE_LOCAL_ROOT}/releases/.staging.XXXXXXXX')
trap 'rm -rf -- \"\$staging\"' EXIT
tar -xf '${remote_archive}' -C \"\$staging\"
printf '%s\\n' '${digest}' > \"\$staging/.fmad-source-sha256\"
chmod -R a-w \"\$staging\"
mv \"\$staging\" \"\$release\"
trap - EXIT
current_after=\$(readlink '${REMOTE_LOCAL_ROOT}/current' || true)
test \"\$current_before\" = \"\$current_after\"
printf 'Current release before staging: %s\\nCurrent release after staging: %s\\n' \"\$current_before\" \"\$current_after\"
echo 'Staged local snapshot without activation: ${digest}'" /workspace
    exit 0
  fi
  container_bash "set -euo pipefail
exec 9>/tmp/fmad-remote-execution.lock
flock -n 9 || { echo 'Experiment/update in progress; code unchanged.' >&2; exit 1; }
test \"\$(sha256sum '${remote_archive}' | cut -d ' ' -f 1)\" = '${digest}'
mkdir -p '${REMOTE_LOCAL_ROOT}/releases'
if [[ ! -d '${REMOTE_LOCAL_ROOT}/releases/${digest}' ]]; then
  staging=\$(mktemp -d '${REMOTE_LOCAL_ROOT}/releases/.staging.XXXXXXXX')
  tar -xf '${remote_archive}' -C \"\$staging\"
  printf '%s\\n' '${digest}' > \"\$staging/.fmad-source-sha256\"
  chmod -R a-w \"\$staging\"
  mv \"\$staging\" '${REMOTE_LOCAL_ROOT}/releases/${digest}'
fi
test ! -e '${REMOTE_LOCAL_ROOT}/current' || test -L '${REMOTE_LOCAL_ROOT}/current'
ln -s '${REMOTE_LOCAL_ROOT}/releases/${digest}' '${REMOTE_LOCAL_ROOT}/.current.'\$\$
mv -Tf '${REMOTE_LOCAL_ROOT}/.current.'\$\$ '${REMOTE_LOCAL_ROOT}/current'
echo 'Activated local snapshot: ${digest}'" /workspace
)

update_code() {
  if [[ "${REMOTE_CODE_MODE}" == local ]]; then
    if [[ "$#" -eq 1 && "$1" == "--stage-only" ]]; then
      update_local_code 1
    elif [[ "$#" -eq 0 ]]; then
      update_local_code
    else
      echo "Usage: scripts/run_remote.sh update-code [--stage-only]" >&2
      return 1
    fi
    return
  fi
  if [[ "$#" -gt 0 ]]; then
    echo "--stage-only is available only with REMOTE_CODE_MODE=local." >&2
    return 1
  fi
  start_container
  container_bash 'set -euo pipefail
exec 9>/tmp/fmad-remote-execution.lock
flock -n 9 || { echo "Experiment/update in progress; code unchanged." >&2; exit 1; }
git rev-parse --is-inside-work-tree >/dev/null
test -z "$(git status --porcelain)" || { echo "Checkout is dirty; refusing pull." >&2; exit 1; }
git pull --ff-only
git rev-parse HEAD'
}

exec_in_container() {
  require_remote_host
  if [[ "$#" -eq 0 || -z "${REMOTE_GPU_DEVICES}" ]]; then
    echo "A command and explicit REMOTE_GPU_DEVICES (prefer GPU UUID) are required." >&2
    return 1
  fi
  if ! is_container_running; then
    echo "Container not running. Starting..." >&2
    start_container
  fi
  local cmd_args
  printf -v cmd_args '%q ' "$@"
  container_bash "set -euo pipefail
exec 9>/tmp/fmad-remote-execution.lock
# Shared lock: several experiments may run at once -- they are isolated by
# CUDA_VISIBLE_DEVICES and by disjoint output roots. Code activation and
# git pull take this same lock exclusively, so no run can start while the
# active source tree is being swapped.
flock -sn 9 || { echo 'Code update in progress; experiment not started.' >&2; exit 1; }
mkdir -p /tmp/fmad-execution-logs
run_log=\$(mktemp -d /tmp/fmad-execution-logs/run.XXXXXXXX)
printf '%s\\n' running > \"\$run_log/status\"
trap 'rc=\$?; printf \"%s\\n\" \"\$rc\" > \"\$run_log/exit_code\"; if [[ \$rc -eq 0 ]]; then echo completed; else echo failed; fi > \"\$run_log/status\"' EXIT
printf '%s\\n' \"Log directory: \$run_log\"
if [[ -f .fmad-source-sha256 ]]; then
  cp .fmad-source-sha256 \"\$run_log/source_sha256\"
else
  git rev-parse HEAD > \"\$run_log/commit\" 2>/dev/null || true
fi
$cmd_args > >(tee \"\$run_log/stdout.log\") 2> >(tee \"\$run_log/stderr.log\" >&2)"
}

show_logs() {
  require_remote_host
  local lines="${1:-50}"
  [[ "${lines}" =~ ^[0-9]+$ ]] || { echo "Log line count must be numeric." >&2; return 1; }
  container_bash "find /tmp/fmad-execution-logs -name '*.log' -type f -exec tail -n ${lines} {} +"
}

show_remote_status() {
  require_remote_host
  echo "=== Container status ==="
  ssh_cmd "${REMOTE_HOST}" "docker ps -a --filter name='${CONTAINER_NAME}' --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}' 2>/dev/null || echo 'No container found'"
  echo ""
  echo "=== GPU ==="
  echo "Requested CUDA_VISIBLE_DEVICES: ${REMOTE_GPU_DEVICES:-UNSET}"
  ssh_cmd "${REMOTE_HOST}" "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv,noheader" || return
  echo ""
  echo "=== Remote results ==="
  ssh_cmd "${REMOTE_HOST}" "ls -la '${REMOTE_RESULTS_DIR}/' 2>/dev/null || echo 'No results directory yet'"
}

pull_results() {
  require_remote_host
  local includes=(--include=metrics.json --include=summary.tsv)
  if [[ "$#" -eq 0 ]]; then
    includes+=(--include=config.yaml --include=run_manifest.json --include=status.json --include=comparison.md)
  elif [[ "$#" -ne 1 || "$1" != --metrics-only ]]; then
    echo "Usage: scripts/run_remote.sh pull [--metrics-only]" >&2
    return 1
  fi
  mkdir -p "${LOCAL_RESULTS_DIR}"
  echo "Pulling results from remote..."
  run_rsync -az --prune-empty-dirs -e "$(rsync_ssh_cmd)" \
    --include='*/' "${includes[@]}" --exclude='*' \
    "$(rsync_remote_target "${REMOTE_RESULTS_DIR}/")" "${LOCAL_RESULTS_DIR}/"
  echo "Results pulled to: ${LOCAL_RESULTS_DIR}"
}

# ── Main ─────────────────────────────────────────────────────────────────

main() {
  if [[ "$#" -lt 1 ]]; then
    usage >&2
    exit 1
  fi

  local mode="$1"
  shift
  validate_remote_repo_dir_chars "${REMOTE_REPO_DIR}"
  for value in "${REMOTE_WORKDIR}" "${REMOTE_RESULTS_DIR}" "${CONTAINER_NAME}" \
    "${REMOTE_LOCAL_ROOT}" "${REMOTE_PYTHON_HOME}" "${REMOTE_TORCH_HOME}"; do
    validate_remote_repo_dir_chars "${value}"
  done
  [[ "${REMOTE_GPU_DEVICES}" =~ ^[A-Za-z0-9_,./-]*$ ]] || { echo "Invalid REMOTE_GPU_DEVICES." >&2; exit 1; }
  [[ "${REMOTE_CODE_MODE}" == git || "${REMOTE_CODE_MODE}" == local ]] || { echo "Invalid REMOTE_CODE_MODE." >&2; exit 1; }
  if [[ "$#" -gt 0 ]]; then
    if [[ "$1" == "--" ]]; then
      shift
    fi
  fi

  case "${mode}" in
    sync)
      sync_repo
      ;;
    sync-data)
      sync_data
      ;;
    build)
      build_image
      ;;
    start)
      start_container
      ;;
    stop)
      stop_container
      ;;
    exec)
      exec_in_container "$@"
      ;;
    run)
      # Convenience: ensure container is running, then exec
      exec_in_container "$@"
      ;;
    update-code)
      update_code "$@"
      ;;
    logs)
      show_logs "${1:-50}"
      ;;
    shell)
      require_remote_host
      if ! is_container_running; then
        start_container
      fi
      echo "Opening shell in '${CONTAINER_NAME}'..."
      ssh_cmd "${REMOTE_HOST}" -t "docker exec -it '${CONTAINER_NAME}' bash"
      ;;
    status)
      show_remote_status
      ;;
    pull)
      pull_results "$@"
      ;;
    setup)
      echo "setup is disabled. Use an existing container; provisioning requires a separate task." >&2
      return 1
      ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
