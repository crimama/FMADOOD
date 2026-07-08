#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASKPASS_SCRIPT="${ROOT_DIR}/scripts/ssh_askpass_env.sh"

REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
JUMP_HOST="${JUMP_HOST:-}"
JUMP_PORT="${JUMP_PORT:-22}"
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
)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_remote.sh <mode> [-- command args...]

Modes (infrastructure):
  sync      - rsync project to remote server
  sync-data - rsync datasets to remote server
  build     - build Docker image on remote server

Modes (persistent container):
  start     - create & start persistent container (idempotent)
  exec      - run a command inside the running container
  stop      - stop and remove the container
  logs      - show container logs (tail)
  shell     - open interactive shell in the container

Modes (convenience):
  run       - start (if needed) + exec command
  status    - show container & results status
  pull      - rsync remote results back to local
  setup     - sync + sync-data + build + start (full first-time setup)

Required env:
  REMOTE_HOST=user@server

Example (first-time setup):
  source configs/remote/147.47.134.98-via-147.47.39.144.env
  scripts/run_remote.sh setup

Example (run experiment — container stays alive):
  scripts/run_remote.sh run -- python3 run.py \
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
  if [[ -n "${SSH_PASSWORD:-}" ]]; then
    printf 'env SSH_PASSWORD=%q DISPLAY=%q SSH_ASKPASS=%q SSH_ASKPASS_REQUIRE=force setsid -w ssh' \
      "${SSH_PASSWORD}" "${DISPLAY:-dummy}" "${ASKPASS_SCRIPT}"
  else
    printf 'ssh'
  fi
  if [[ -n "${JUMP_HOST}" ]]; then
    printf ' -J %q:%q -p %q' "${JUMP_HOST}" "${JUMP_PORT}" "${REMOTE_PORT}"
  else
    printf ' -p %q' "${REMOTE_PORT}"
  fi
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
    rsync -az -e "$(rsync_ssh_cmd)" "${resolved_local_data_root}/${path}" "$(rsync_remote_target "${REMOTE_DATA_ROOT}/")"
  done
  echo "Data sync complete."
}

sync_repo() {
  require_remote_host
  mkdir -p "${LOCAL_RESULTS_DIR}"
  local delete_flag=()
  if [[ "${SYNC_DELETE}" == "1" ]]; then
    delete_flag=(--delete)
  fi
  ssh_cmd "${REMOTE_HOST}" "mkdir -p '${REMOTE_REPO_DIR}' '${REMOTE_RESULTS_DIR}' '${REMOTE_CACHE_ROOT}/torch/hub' '${REMOTE_CACHE_ROOT}/torch/checkpoints' '${REMOTE_CACHE_ROOT}/huggingface' '${REMOTE_CACHE_ROOT}/pip'"
  echo "Syncing project to remote..."
  rsync -az -e "$(rsync_ssh_cmd)" "${delete_flag[@]}" \
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

  # Remove stopped container if it exists
  if is_container_exists; then
    echo "Removing stopped container '${CONTAINER_NAME}'..."
    ssh_cmd "${REMOTE_HOST}" "docker rm '${CONTAINER_NAME}'" >/dev/null
  fi

  local extra_args="${EXTRA_DOCKER_ARGS:---gpus all --shm-size=4g}"
  local cache_env="-e TORCH_HOME=${REMOTE_WORKDIR}/.cache/torch -e HF_HOME=${REMOTE_WORKDIR}/.cache/huggingface -e PIP_CACHE_DIR=${REMOTE_WORKDIR}/.cache/pip"
  local cache_mounts="-v '${REMOTE_CACHE_ROOT}/torch:${REMOTE_WORKDIR}/.cache/torch' -v '${REMOTE_CACHE_ROOT}/huggingface:${REMOTE_WORKDIR}/.cache/huggingface' -v '${REMOTE_CACHE_ROOT}/pip:${REMOTE_WORKDIR}/.cache/pip'"

  echo "Starting persistent container '${CONTAINER_NAME}'..."
  ssh_cmd "${REMOTE_HOST}" "cd '${REMOTE_REPO_DIR}' && mkdir -p '${REMOTE_RESULTS_DIR}' && docker run -d --name '${CONTAINER_NAME}' ${extra_args} \
    -v '${REMOTE_REPO_DIR}:${REMOTE_WORKDIR}' \
    -v '${REMOTE_DATA_ROOT}:${REMOTE_DATA_ROOT}' \
    ${cache_mounts} \
    ${cache_env} \
    -w '${REMOTE_WORKDIR}' \
    '${IMAGE_NAME}' sleep infinity"
  echo "Container started."
}

stop_container() {
  require_remote_host
  if is_container_running; then
    echo "Stopping container '${CONTAINER_NAME}'..."
    ssh_cmd "${REMOTE_HOST}" "docker stop '${CONTAINER_NAME}'" >/dev/null
  fi
  if is_container_exists; then
    echo "Removing container '${CONTAINER_NAME}'..."
    ssh_cmd "${REMOTE_HOST}" "docker rm '${CONTAINER_NAME}'" >/dev/null
  fi
  echo "Container stopped and removed."
}

exec_in_container() {
  require_remote_host
  if ! is_container_running; then
    echo "Container not running. Starting..." >&2
    start_container
  fi
  local cmd_args="$*"
  echo "Executing: ${cmd_args}"
  ssh_cmd "${REMOTE_HOST}" "docker exec '${CONTAINER_NAME}' bash -lc '${cmd_args}'"
}

show_logs() {
  require_remote_host
  local lines="${1:-50}"
  ssh_cmd "${REMOTE_HOST}" "docker logs --tail ${lines} '${CONTAINER_NAME}' 2>&1 | tr '\r' '\n' | grep -v '^\$' | tail -${lines}"
}

show_remote_status() {
  require_remote_host
  echo "=== Container status ==="
  ssh_cmd "${REMOTE_HOST}" "docker ps -a --filter name='${CONTAINER_NAME}' --format 'table {{.Names}}\t{{.Status}}\t{{.RunningFor}}' 2>/dev/null || echo 'No container found'"
  echo ""
  echo "=== GPU ==="
  local gpu_id
  gpu_id="$(echo "${EXTRA_DOCKER_ARGS:-}" | grep -oP 'device=\K[0-9]+' || echo '')"
  if [[ -n "${gpu_id}" ]]; then
    ssh_cmd "${REMOTE_HOST}" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader -i ${gpu_id}" 2>/dev/null || true
  else
    ssh_cmd "${REMOTE_HOST}" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null | head -3 || true
  fi
  echo ""
  echo "=== Remote results ==="
  ssh_cmd "${REMOTE_HOST}" "ls -la '${REMOTE_RESULTS_DIR}/' 2>/dev/null || echo 'No results directory yet'"
}

pull_results() {
  require_remote_host
  mkdir -p "${LOCAL_RESULTS_DIR}"
  echo "Pulling results from remote..."
  rsync -az -e "$(rsync_ssh_cmd)" "$(rsync_remote_target "${REMOTE_RESULTS_DIR}/")" "${LOCAL_RESULTS_DIR}/"
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
  local run_args=""
  if [[ "$#" -gt 0 ]]; then
    if [[ "$1" == "--" ]]; then
      shift
    fi
    run_args="$*"
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
      exec_in_container "${run_args}"
      ;;
    run)
      # Convenience: ensure container is running, then exec
      start_container
      exec_in_container "${run_args}"
      ;;
    logs)
      show_logs "${run_args:-50}"
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
      pull_results
      ;;
    setup)
      sync_repo
      sync_data
      build_image
      start_container
      echo ""
      echo "Setup complete. Run experiments with:"
      echo "  scripts/run_remote.sh exec -- python3 run.py --config configs/default.yaml --data-root ${REMOTE_DATA_ROOT}/mvtec_ad_2 --output-dir /workspace/results_remote/exp01"
      ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
