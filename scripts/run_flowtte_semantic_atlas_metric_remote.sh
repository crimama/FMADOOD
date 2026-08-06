#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${ROOT_DIR}/configs/remote/dsba5.env"
ASKPASS_SCRIPT="${ROOT_DIR}/scripts/ssh_askpass_env.sh"
FIXED_CONTAINER="hun_fsad_tta_012"
HOST_GPU_ID="${HOST_GPU_ID:-3}"
CONTAINER_CUDA_SLOT="0"
REMOTE_SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT:-/workspace/isolated_runs}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/workspace/results_remote}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
SUPPORT_REL="skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json"
SUPPORT_SHA256="9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
LOCAL_PULL_ROOT="${LOCAL_PULL_ROOT:-${ROOT_DIR}/results/remote_runs/dsba5}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-512}"
GPU_IDLE_UTILIZATION="${GPU_IDLE_UTILIZATION:-5}"
SMOKE_OBJECTS="fruit_jelly,walnuts,vial,fabric"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

require_run_name() {
  [[ -n "${RUN_NAME:-}" ]] || die "RUN_NAME is required for ${MODE}"
  [[ "${RUN_NAME}" =~ ^flowtte_semantic_atlas_[A-Za-z0-9._-]+$ ]] || die "unsafe semantic-atlas RUN_NAME"
}

load_preset() {
  [[ -f "${PRESET}" ]] || die "missing preset: ${PRESET}"
  # shellcheck disable=SC1090
  source "${PRESET}"
  [[ -n "${REMOTE_HOST:-}" && -n "${REMOTE_PORT:-}" ]] || die "incomplete preset"
}

ssh_remote() {
  local args=(-o StrictHostKeyChecking=no -p "${REMOTE_PORT}")
  if [[ -n "${JUMP_HOST:-}" ]]; then args=(-J "${JUMP_HOST}:${JUMP_PORT}" "${args[@]}"); fi
  if [[ -n "${SSH_PASSWORD:-}" ]]; then
    env SSH_PASSWORD="${SSH_PASSWORD}" DISPLAY="${DISPLAY:-dummy}" \
      SSH_ASKPASS="${ASKPASS_SCRIPT}" SSH_ASKPASS_REQUIRE=force \
      setsid -w ssh "${args[@]}" "$@"
  else
    ssh "${args[@]}" "$@"
  fi
}

assert_container() {
  ssh_remote "${REMOTE_HOST}" \
    "test \"\$(docker inspect -f '{{.State.Running}}' '${FIXED_CONTAINER}' 2>/dev/null)\" = true"
}

assert_gpu_binding_and_idle() {
  ssh_remote "${REMOTE_HOST}" bash -s -- \
    "${FIXED_CONTAINER}" "${HOST_GPU_ID}" "${CONTAINER_CUDA_SLOT}" \
    "${GPU_IDLE_MEMORY_MIB}" "${GPU_IDLE_UTILIZATION}" <<'REMOTE'
set -euo pipefail
container="$1"; host_gpu="$2"; container_slot="$3"; memory_limit="$4"; util_limit="$5"
host_uuid="$(nvidia-smi -i "${host_gpu}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
container_uuid="$(docker exec "${container}" nvidia-smi -i "${container_slot}" \
  --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
[[ -n "${host_uuid}" && "${host_uuid}" == "${container_uuid}" ]]
sample="$(nvidia-smi -i "${host_gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
memory="$(cut -d, -f1 <<<"${sample}" | tr -d '[:space:]')"
utilization="$(cut -d, -f2 <<<"${sample}" | tr -d '[:space:]')"
(( memory <= memory_limit && utilization <= util_limit )) || exit 73
processes="$(nvidia-smi -i "${host_gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
! grep -Eq '[0-9]+' <<<"${processes}" || exit 74
printf 'GPU_BINDING_OK host_gpu=%s container_slot=%s uuid=%s memory_mib=%s utilization=%s\n' \
  "${host_gpu}" "${container_slot}" "${host_uuid}" "${memory}" "${utilization}"
REMOTE
}

make_archive() {
  tar --exclude=.git --exclude=.omx --exclude=.venv --exclude=results \
    --exclude=results_remote --exclude=configs/remote --exclude=data \
    --exclude='__pycache__' --exclude='*.pyc' -C "${ROOT_DIR}" -cf "$1" .
}

stream_snapshot() {
  local archive="$1" hash="$2" snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  ssh_remote "${REMOTE_HOST}" \
    "docker exec -i '${FIXED_CONTAINER}' bash -lc '
      set -euo pipefail
      test ! -e \"${snapshot}\"
      mkdir -p \"${snapshot}\"
      tee /tmp/${RUN_NAME}.tar >/dev/null
      test \"\$(sha256sum /tmp/${RUN_NAME}.tar | awk \"{print \\\$1}\")\" = \"${hash}\"
      tar -C \"${snapshot}\" -xf /tmp/${RUN_NAME}.tar
      rm -f /tmp/${RUN_NAME}.tar
      printf \"archive_sha256=%s\\n\" \"${hash}\" >\"${snapshot}/snapshot_hash.txt\"
    '" <"${archive}"
}

preflight() {
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local core_hash runner_hash aggregate_hash controller_hash
  core_hash="$(sha256sum "${ROOT_DIR}/src/flow_tte/semantic_atlas_metric.py" | awk '{print $1}')"
  runner_hash="$(sha256sum "${ROOT_DIR}/scripts/run_flowtte_semantic_atlas_metric.py" | awk '{print $1}')"
  aggregate_hash="$(sha256sum "${ROOT_DIR}/scripts/aggregate_flowtte_semantic_atlas_metric.py" | awk '{print $1}')"
  controller_hash="$(sha256sum "${ROOT_DIR}/scripts/run_flowtte_semantic_atlas_metric_remote.sh" | awk '{print $1}')"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    test ! -e \"${root}\"
    mkdir -p '${root}'
    exec > >(tee -a '${root}/preflight_console.log') 2>&1
    echo step=container_imports
    test \"\$(python3 -c \"import torch; print(torch.cuda.device_count())\")\" -eq 1
    python3 -c \"import cv2,numpy,scipy,sklearn,tifffile,torch; from PIL import Image\"
    PYTHONPATH='${snapshot}:${snapshot}/src:${snapshot}/scripts' python3 -c \
      \"import flow_tte.semantic_atlas_metric; import run_flowtte_semantic_atlas_metric; import aggregate_flowtte_semantic_atlas_metric\"
    free_bytes=\$(df -PB1 /workspace | awk \"NR==2 {print \\\$4}\")
    test \"\${free_bytes}\" -ge $((MIN_FREE_GIB * 1024 * 1024 * 1024))
    test \"\$(sha256sum '${snapshot}/${SUPPORT_REL}' | awk \"{print \\\$1}\")\" = '${SUPPORT_SHA256}'
    test \"\$(sha256sum '${snapshot}/src/flow_tte/semantic_atlas_metric.py' | awk \"{print \\\$1}\")\" = '${core_hash}'
    test \"\$(sha256sum '${snapshot}/scripts/run_flowtte_semantic_atlas_metric.py' | awk \"{print \\\$1}\")\" = '${runner_hash}'
    test \"\$(sha256sum '${snapshot}/scripts/aggregate_flowtte_semantic_atlas_metric.py' | awk \"{print \\\$1}\")\" = '${aggregate_hash}'
    test \"\$(sha256sum '${snapshot}/scripts/run_flowtte_semantic_atlas_metric_remote.sh' | awk \"{print \\\$1}\")\" = '${controller_hash}'
    echo step=data_and_compile
    for object_name in fruit_jelly walnuts vial fabric; do
      test -d '${DATA_ROOT}/'\"\${object_name}\"'/train/good'
    done
    python3 -m py_compile \
      '${snapshot}/src/flow_tte/semantic_atlas_metric.py' \
      '${snapshot}/scripts/run_flowtte_semantic_atlas_metric.py' \
      '${snapshot}/scripts/aggregate_flowtte_semantic_atlas_metric.py'
    if python3 -c \"import pytest\" >/dev/null 2>&1; then
      (
        cd '${snapshot}'
        PYTHONPATH='${snapshot}:${snapshot}/src:${snapshot}/scripts' python3 -m pytest -q \
          tests/test_semantic_atlas_metric.py \
          tests/test_flowtte_semantic_atlas_metric_runner.py \
          tests/test_aggregate_flowtte_semantic_atlas_metric.py \
          tests/test_run_flowtte_semantic_atlas_metric_remote.py \
          >'/tmp/${RUN_NAME}_pytest.log'
      )
    else
      echo 'pytest_unavailable_runtime_smoke_used' >'/tmp/${RUN_NAME}_pytest.log'
    fi
    smoke_dir=/tmp/${RUN_NAME}_smoke
    test ! -e \"\${smoke_dir}\"
    mkdir -p \"\${smoke_dir}\"
    PYTHONPATH='${snapshot}:${snapshot}/src:${snapshot}/scripts' \
      python3 '${snapshot}/scripts/run_flowtte_semantic_atlas_metric.py' \
        --phase smoke --output-root \"\${smoke_dir}\" \
        --objects '${SMOKE_OBJECTS}' --device cpu \
        >'/tmp/${RUN_NAME}_runner_smoke.log'
    test -f \"\${smoke_dir}/run_manifest.json\"
    echo step=dinov3_cache
    TORCH_HOME=/workspace/.cache/torch FMAD_DINOV3_OFFLINE=1 \
      PYTHONPATH='${snapshot}:${snapshot}/src:${snapshot}/scripts' \
      python3 '${snapshot}/scripts/preflight_dinov3_cache.py' \
        --model dinov3_vith16plus --device cpu --feature-layers 7,15,23,31 \
      >'/tmp/${RUN_NAME}_dinov3.log'
    rm -rf -- \"\${smoke_dir}\"
    cp '${snapshot}/snapshot_hash.txt' '${root}/snapshot_hash.txt'
    cp '/tmp/${RUN_NAME}_pytest.log' '${root}/preflight_pytest.log'
    cp '/tmp/${RUN_NAME}_runner_smoke.log' '${root}/preflight_runner_smoke.log'
    cp '/tmp/${RUN_NAME}_dinov3.log' '${root}/preflight_dinov3.log'
    nvidia-smi --query-gpu=index,uuid,memory.total --format=csv,noheader >'${root}/gpu_visibility.txt'
    {
      echo 'status=passed'
      echo 'core_sha256=${core_hash}'
      echo 'runner_sha256=${runner_hash}'
      echo 'aggregator_sha256=${aggregate_hash}'
      echo 'controller_sha256=${controller_hash}'
      echo 'support_manifest_sha256=${SUPPORT_SHA256}'
      echo 'host_gpu_id=${HOST_GPU_ID}'
      echo 'container_cuda_slot=${CONTAINER_CUDA_SLOT}'
      echo 'container_cuda_count=1'
      echo \"free_bytes=\${free_bytes}\"
    } >'${root}/preflight.txt'
  '"
}

run_mechanism() {
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="${CONTAINER_CUDA_SLOT}" \
    TORCH_HOME=/workspace/.cache/torch FMAD_DINOV3_OFFLINE=1 \
    PYTHONPATH="${snapshot}:${snapshot}/src:${snapshot}/scripts" \
    python3 "${snapshot}/scripts/run_flowtte_semantic_atlas_metric.py" \
      --phase mechanism --data-root "${DATA_ROOT}" \
      --support-json "${snapshot}/${SUPPORT_REL}" \
      --output-root "${root}/mechanism" --objects "${SMOKE_OBJECTS}" \
      --device cuda:0 --seed 0 --host-gpu-id "${HOST_GPU_ID}" \
      --container-cuda-slot "${CONTAINER_CUDA_SLOT}" \
      >"${root}/mechanism.log" 2>&1
}

aggregate_mechanism() {
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  PYTHONPATH="${snapshot}:${snapshot}/src:${snapshot}/scripts" python3 \
    "${snapshot}/scripts/aggregate_flowtte_semantic_atlas_metric.py" \
      --phase mechanism --output-root "${root}/mechanism/aggregate" \
      --summaries \
        "${root}/mechanism/fruit_jelly/summary.json" \
        "${root}/mechanism/walnuts/summary.json" \
        "${root}/mechanism/vial/summary.json" \
        "${root}/mechanism/fabric/summary.json" \
      >"${root}/mechanism_aggregate.log" 2>&1
}

write_compact_checksums() {
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  (
    cd "${root}"
    find . -type f \
      ! -path './compact_artifact_sha256.txt' \
      ! -path './dense_feature_cache/*' \
      ! -path './patch_score_cache/*' \
      ! -path './transformed_images/*' \
      ! -path './anomaly_maps/*' \
      ! -path './score_maps/*' \
      -printf '%P\0' | sort -z | xargs -0 sha256sum >compact_artifact_sha256.txt
  )
  if [[ -f "${root}/mechanism/aggregate/all8_support_authorized.txt" ]]; then
    grep -q 'all8_support_only' "${root}/mechanism/aggregate/all8_support_authorized.txt"
  fi
}

internal_controller() {
  require_run_name
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  mkdir -p "${root}"
  trap 'code=$?; if ((code!=0)) && [[ ! -f "'"${root}"'/failed.txt" ]]; then printf "exit_code=%s\n" "${code}" >"'"${root}"'/failed.txt"; fi' EXIT
  exec 9>"${REMOTE_SNAPSHOT_ROOT}/.flowtte_semantic_atlas_gpu3.lock"
  flock -n 9 || { echo cooperative_lock_busy >"${root}/failed.txt"; return 75; }
  printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" >"${root}/started.txt"
  run_mechanism
  aggregate_mechanism
  cp "${root}/mechanism/aggregate/verdict.json" "${root}/protocol_verdict.json"
  write_compact_checksums
  printf 'status=complete_support_only\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" >"${root}/complete.txt"
}

start_remote() {
  require_run_name
  assert_container
  assert_gpu_binding_and_idle
  local archive hash
  archive="$(mktemp "${TMPDIR:-/tmp}/flowtte-semantic-atlas-source.XXXXXX.tar")"
  trap 'rm -f -- "'"${archive}"'"' EXIT
  make_archive "${archive}"
  hash="$(sha256sum "${archive}" | awk '{print $1}')"
  stream_snapshot "${archive}" "${hash}"
  preflight
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    snapshot=${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo
    nohup env RUN_NAME=${RUN_NAME} REMOTE_SNAPSHOT_ROOT=${REMOTE_SNAPSHOT_ROOT} \
      REMOTE_RESULTS_ROOT=${REMOTE_RESULTS_ROOT} DATA_ROOT=${DATA_ROOT} \
      TORCH_HOME=/workspace/.cache/torch FMAD_DINOV3_OFFLINE=1 \
      bash \"\${snapshot}/scripts/run_flowtte_semantic_atlas_metric_remote.sh\" internal-controller \
      >\"\${root}/controller.log\" 2>&1 </dev/null &
    printf \"%s\\n\" \"\$!\" >\"\${root}/controller.pid\"
    printf \"STARTED run_name=%s pid=%s\\n\" '${RUN_NAME}' \"\$!\"
  '"
  rm -f "${archive}"; trap - EXIT
}

status_remote() {
  require_run_name
  assert_container
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    test -d \"\${root}\" && echo result_root=present || echo result_root=missing
    find \"\${root}\" -maxdepth 2 -type f -printf \"artifact=%P bytes=%s\\n\" 2>/dev/null | sort || true
    test -f \"\${root}/complete.txt\" && status=complete || \
      { test -f \"\${root}/failed.txt\" && status=failed || status=running_or_pending; }
    echo run_name=${RUN_NAME}; echo status=\"\${status}\"
    if test -f \"\${root}/controller.pid\"; then
      controller_pid=\$(tr -cd 0-9 <\"\${root}/controller.pid\")
      echo controller_pid=\"\${controller_pid}\"
      if test -n \"\${controller_pid}\" && kill -0 \"\${controller_pid}\" 2>/dev/null; then
        ps -o pid,ppid,stat,etime,cmd -p \"\${controller_pid}\"
        child_pids=\$(pgrep -P \"\${controller_pid}\" | tr '\n' ',' | sed 's/,$//' || true)
        test -z \"\${child_pids}\" || ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p \"\${child_pids}\"
      else
        echo controller_process=not_running
      fi
    fi
    echo summaries=\"\$(find \"\${root}/mechanism\" -name summary.json 2>/dev/null | wc -l)\"
    test -f \"\${root}/mechanism/aggregate/verdict.json\" && cat \"\${root}/mechanism/aggregate/verdict.json\" || true
    test -f \"\${root}/failed.txt\" && cat \"\${root}/failed.txt\" || true
    echo preflight_manifest
    test -f "\${root}/preflight.txt" && cat "\${root}/preflight.txt" || true
    echo progress_tail
    test -f \"\${root}/mechanism/progress.jsonl\" && tail -n 12 \"\${root}/mechanism/progress.jsonl\" || true
    echo preflight_console_tail
    test -f \"\${root}/preflight_console.log\" && tail -n 24 \"\${root}/preflight_console.log\" || true
    echo controller_log_tail
    test -f \"\${root}/controller.log\" && tail -n 20 \"\${root}/controller.log\" || true
    echo mechanism_log_tail
    test -f \"\${root}/mechanism.log\" && tail -n 16 \"\${root}/mechanism.log\" || true
  '"
  ssh_remote "${REMOTE_HOST}" \
    "nvidia-smi -i ${HOST_GPU_ID} --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; \
     nvidia-smi -i ${HOST_GPU_ID} --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits; \
     gpu_pids=\$(nvidia-smi -i ${HOST_GPU_ID} --query-compute-apps=pid --format=csv,noheader,nounits | paste -sd, -); \
     test -z \"\${gpu_pids}\" || ps -o pid,ppid,etime,cmd -p \"\${gpu_pids}\""
}

pull_remote() {
  require_run_name
  assert_container
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}"
  local partial="${LOCAL_PULL_ROOT}/${RUN_NAME}.partial.$$"
  [[ ! -e "${destination}" ]] || die "destination exists: ${destination}"
  mkdir -p "${partial}"
  trap 'rm -rf -- "'"${partial}"'"' EXIT
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    test -f '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/complete.txt'
    tar -C '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' -cf - .
  '" | tar -C "${partial}" -xf -
  test -f "${partial}/complete.txt"
  # Older completed runs may contain an impossible self-checksum entry because
  # the manifest was opened before find(1) enumerated it.  Verify every payload
  # artifact while excluding only that manifest self-reference.
  (
    cd "${partial}"
    sha256sum -c <(grep -vE '  compact_artifact_sha256\.txt$' compact_artifact_sha256.txt)
  )
  mv "${partial}" "${destination}"; trap - EXIT
  printf 'PULLED %s\n' "${destination}"
}

cleanup_remote() {
  require_run_name
  assert_container
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}"
  [[ "${snapshot}" == /workspace/isolated_runs/flowtte_semantic_atlas_* ]] || die "unsafe cleanup"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    test -f '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/complete.txt' -o \
      -f '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/invalidated.txt'
    rm -rf -- '${snapshot}'
    test ! -e '${snapshot}'
  '"
}

invalidate_remote() {
  require_run_name
  assert_container
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    test -f \"\${root}/controller.pid\"
    controller_pid=\$(tr -cd 0-9 <\"\${root}/controller.pid\")
    [[ -n \"\${controller_pid}\" ]]
    terminate_tree() {
      local parent=\"\$1\" child
      while read -r child; do
        [[ -n \"\${child}\" ]] || continue
        terminate_tree \"\${child}\"
        kill -TERM \"\${child}\" 2>/dev/null || true
      done < <(pgrep -P \"\${parent}\" || true)
    }
    terminate_tree \"\${controller_pid}\"
    kill -TERM \"\${controller_pid}\" 2>/dev/null || true
    printf 'reason=implementation_changed_before_scientific_result\\ninvalidated_utc=%s\\n' \
      \"\$(date -u +%FT%TZ)\" >\"\${root}/invalidated.txt\"
  '"
}

case "${MODE}" in
  start)
    RUN_NAME="${RUN_NAME:-flowtte_semantic_atlas_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
    load_preset; start_remote
    ;;
  status) load_preset; status_remote ;;
  pull) load_preset; pull_remote ;;
  cleanup) load_preset; cleanup_remote ;;
  invalidate) load_preset; invalidate_remote ;;
  internal-controller) internal_controller ;;
  *) die "usage: RUN_NAME=flowtte_semantic_atlas_<id> $0 {start|status|pull|cleanup|invalidate}" ;;
esac
