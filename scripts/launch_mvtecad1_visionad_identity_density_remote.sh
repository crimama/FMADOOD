#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-start}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-mvtecad1_visionad_identity_density_s1_2_4_8_16_20260714_v1}"
FEATURE_LAYERS="${FEATURE_LAYERS:-2,3,4,5,6,7,8,9}"
WORKER_SCRIPT="${WORKER_SCRIPT:-run_mvtecad1_visionad_identity_density_sharded.sh}"
EVALUATE_EXISTING="${EVALUATE_EXISTING:-0}"
PULL_COMPACT="${PULL_COMPACT:-0}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/fsad_tta}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/workspace/results_remote}"
ASKPASS_SCRIPT="${ROOT_DIR}/scripts/ssh_askpass_env.sh"
OBJECTS=(bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper)

load_preset() { source "${ROOT_DIR}/configs/remote/$1.env"; }

ssh_remote() {
  local args=(-p "${REMOTE_PORT}")
  [[ -n "${JUMP_HOST:-}" ]] && args=(-J "${JUMP_HOST}:${JUMP_PORT}" "${args[@]}")
  if [[ -n "${SSH_PASSWORD:-}" ]]; then
    env SSH_PASSWORD="${SSH_PASSWORD}" SSH_ASKPASS="${ASKPASS_SCRIPT}" SSH_ASKPASS_REQUIRE=force \
      DISPLAY=dummy setsid -w ssh "${args[@]}" "$@"
  else
    ssh "${args[@]}" "$@"
  fi
}

sync_repo() {
  tar --exclude='__pycache__' -C "${ROOT_DIR}" -cf - scripts src fmad visionad_density \
    | ssh_remote "${REMOTE_HOST}" "docker exec -i '${CONTAINER_NAME}' tar -C '${REMOTE_REPO}' -xf -"
  ssh_remote "${REMOTE_HOST}" "docker exec '${CONTAINER_NAME}' cp '${REMOTE_REPO}/src/backbones.py' /workspace/src/backbones.py"
  ssh_remote "${REMOTE_HOST}" "docker exec '${CONTAINER_NAME}' cp '${REMOTE_REPO}/src/dinov2_loader.py' /workspace/src/dinov2_loader.py"
}

data_root() {
  case "$1" in
    dsba3) echo /workspace/data/MVTecAD ;;
    dsba5) echo /home/woojun/dataset/mvtec_ad ;;
  esac
}

gpu_count() { [[ "$1" == dsba3 ]] && echo 4 || echo 2; }

preflight() {
  local tag="$1" root="$2" count="$3"
  ssh_remote "${REMOTE_HOST}" "docker exec '${CONTAINER_NAME}' bash -lc '
    set -euo pipefail
    test -d \"${root}\"
    test \"\$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)\" -eq ${count}
    for object in ${OBJECTS[*]}; do
      test -n \"\$(find \"${root}/\${object}/train/good\" -maxdepth 1 -type f -print -quit)\"
      test -d \"${root}/\${object}/test\"
    done
    TORCH_HOME=/workspace/.cache/torch FMAD_DINOV2_OFFLINE=1 PYTHONPATH=${REMOTE_REPO} \
      python3 ${REMOTE_REPO}/scripts/preflight_dinov2_cache.py --model dinov2_vitb14_reg \
      >/tmp/${RUN_NAME}_${tag}_dinov2_preflight.log
    echo PREFLIGHT_OK host=${tag} data_root=${root} gpu_count=${count}
  '"
}

start_host() {
  local tag="$1" root count
  load_preset "${tag}"
  root="$(data_root "${tag}")"; count="$(gpu_count "${tag}")"
  echo "SYNC host=${tag}"
  sync_repo
  echo "PREFLIGHT host=${tag}"
  preflight "${tag}" "${root}" "${count}"
  ssh_remote "${REMOTE_HOST}" "docker exec '${CONTAINER_NAME}' bash -lc '
    set -euo pipefail
    out=${REMOTE_RESULTS_ROOT}/${RUN_NAME}; mkdir -p \"\${out}\"
    if test -f \"\${out}/${tag}_complete.txt\"; then echo ALREADY_COMPLETE host=${tag}; exit 0; fi
    if test -s \"\${out}/${tag}_controller.pid\" \
      && kill -0 \"\$(cat \"\${out}/${tag}_controller.pid\")\" 2>/dev/null \
      && ! ps -o stat= -p \"\$(cat \"\${out}/${tag}_controller.pid\")\" | grep -q \"^[[:space:]]*Z\"; then
      echo ALREADY_RUNNING host=${tag}; exit 0
    fi
    nohup env FSAD_ROOT=${REMOTE_REPO} DATA_ROOT=${root} RESULTS_ROOT=${REMOTE_RESULTS_ROOT} \
      RUN_NAME=${RUN_NAME} HOST_TAG=${tag} FEATURE_LAYERS=${FEATURE_LAYERS} EVALUATE_EXISTING=${EVALUATE_EXISTING} \
      bash ${REMOTE_REPO}/scripts/${WORKER_SCRIPT} \
      >\"\${out}/${tag}_controller.log\" 2>&1 </dev/null &
    echo \$! >\"\${out}/${tag}_controller.pid\"
    echo STARTED host=${tag} pid=\$!
  '"
}

status_host() {
  local tag="$1"
  load_preset "${tag}"
  ssh_remote "${REMOTE_HOST}" "docker exec '${CONTAINER_NAME}' bash -lc '
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}; echo HOST=${tag}
    test -f \"\${root}/${tag}_complete.txt\" && echo COMPLETE=yes || echo COMPLETE=no
    echo MANIFESTS=\$(find \"\${root}\" -name run_manifest.json 2>/dev/null | wc -l)
    echo ACTIVE=\$(ps -eo args= | grep -E \"run_flow_tte_mvtec_ad1.py|visionad_density.run_mvtec\" | grep -v grep | wc -l)
    test -s \"\${root}/${tag}_controller.pid\" && ps -o pid=,stat=,etime=,args= -p \"\$(cat \"\${root}/${tag}_controller.pid\")\" || true
    echo ERRORS=\$(grep -R -l -E \"Traceback|CUDA out of memory|RuntimeError:\" \"\${root}\" 2>/dev/null | wc -l)
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  '"
}

pull_host() {
  local tag="$1" local_root
  local_root="${ROOT_DIR}/results/remote_runs/${tag}/${RUN_NAME}"
  load_preset "${tag}"
  mkdir -p "${local_root}"
  local excludes=()
  test "${PULL_COMPACT}" = 1 && excludes+=(--exclude='*/anomaly_maps')
  ssh_remote "${REMOTE_HOST}" \
    "docker exec '${CONTAINER_NAME}' tar ${excludes[*]} -C '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' -cf - ." \
    | tar -C "${local_root}" -xf -
  echo "PULLED host=${tag} root=${local_root}"
}

case "${MODE}" in
  start) start_host dsba3; start_host dsba5 ;;
  status) status_host dsba3; status_host dsba5 ;;
  pull) pull_host dsba3; pull_host dsba5 ;;
  *) echo "usage: $0 {start|status|pull}" >&2; exit 2 ;;
esac
