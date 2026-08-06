#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${ROOT_DIR}/configs/remote/dsba3.env"
ASKPASS_SCRIPT="${ROOT_DIR}/scripts/ssh_askpass_env.sh"
FIXED_CONTAINER="hun_fsad_tta_012"
REMOTE_SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT:-/workspace/isolated_runs}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/workspace/results_remote}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
SUPPORT_REL="skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json"
SUPPORT_SHA256="9f223f7cdffd4defe37aced6edab98aad4e6e133af6c13df22f5b21be81d18a5"
LOCAL_PULL_ROOT="${LOCAL_PULL_ROOT:-${ROOT_DIR}/results/remote_runs/dsba3}"
MIN_FREE_GIB="${MIN_FREE_GIB:-20}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-512}"
GPU_IDLE_UTILIZATION="${GPU_IDLE_UTILIZATION:-5}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

require_run_name() {
  [[ -n "${RUN_NAME:-}" ]] || die "RUN_NAME is required for ${MODE}"
  [[ "${RUN_NAME}" =~ ^gb_cmf_stage_minus1a_[A-Za-z0-9._-]+$ ]] || die "unsafe GB-CMF RUN_NAME"
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

gpu_idle_check() {
  ssh_remote "${REMOTE_HOST}" bash -s -- \
    "${GPU_IDLE_MEMORY_MIB}" "${GPU_IDLE_UTILIZATION}" <<'REMOTE'
set -euo pipefail
memory_limit="$1"; util_limit="$2"
for gpu in 0 1 2 3; do
  sample="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  memory="$(cut -d, -f1 <<<"${sample}" | tr -d '[:space:]')"
  utilization="$(cut -d, -f2 <<<"${sample}" | tr -d '[:space:]')"
  (( memory <= memory_limit && utilization <= util_limit )) || exit 73
  processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  ! grep -Eq '[0-9]+' <<<"${processes}" || exit 74
  printf 'GPU_IDLE gpu=%s memory_mib=%s utilization=%s\n' "${gpu}" "${memory}" "${utilization}"
done
REMOTE
}

make_archive() {
  tar --exclude=.git --exclude=.omx --exclude=.venv --exclude=results \
    --exclude=configs/remote --exclude='__pycache__' --exclude='*.pyc' \
    -C "${ROOT_DIR}" -cf "$1" .
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
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local runner_hash core_hash pipeline_hash aggregate_hash controller_hash
  runner_hash="$(sha256sum "${ROOT_DIR}/scripts/run_gb_cmf_stage_minus1a.py" | awk '{print $1}')"
  core_hash="$(sha256sum "${ROOT_DIR}/src/flow_tte/gb_cmf.py" | awk '{print $1}')"
  pipeline_hash="$(sha256sum "${ROOT_DIR}/src/flow_tte/pipeline.py" | awk '{print $1}')"
  aggregate_hash="$(sha256sum "${ROOT_DIR}/scripts/aggregate_gb_cmf_stage_minus1a.py" | awk '{print $1}')"
  controller_hash="$(sha256sum "${ROOT_DIR}/scripts/run_gb_cmf_stage_minus1a_remote.sh" | awk '{print $1}')"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    test ! -e \"${result_root}\"
    python3 -c \"import cv2,numpy,sklearn,torch\"
    test \"\$(python3 -c \"import torch; print(torch.cuda.device_count())\")\" -eq 4
    free_bytes=\$(df -PB1 /workspace | awk \"NR==2 {print \\\$4}\")
    test \"\${free_bytes}\" -ge $((MIN_FREE_GIB * 1024 * 1024 * 1024))
    test \"\$(sha256sum '${snapshot}/${SUPPORT_REL}' | awk \"{print \\\$1}\")\" = '${SUPPORT_SHA256}'
    test \"\$(sha256sum '${snapshot}/scripts/run_gb_cmf_stage_minus1a.py' | awk \"{print \\\$1}\")\" = '${runner_hash}'
    test \"\$(sha256sum '${snapshot}/src/flow_tte/gb_cmf.py' | awk \"{print \\\$1}\")\" = '${core_hash}'
    test \"\$(sha256sum '${snapshot}/src/flow_tte/pipeline.py' | awk \"{print \\\$1}\")\" = '${pipeline_hash}'
    test \"\$(sha256sum '${snapshot}/scripts/aggregate_gb_cmf_stage_minus1a.py' | awk \"{print \\\$1}\")\" = '${aggregate_hash}'
    test \"\$(sha256sum '${snapshot}/scripts/run_gb_cmf_stage_minus1a_remote.sh' | awk \"{print \\\$1}\")\" = '${controller_hash}'
    python3 -m py_compile '${snapshot}/src/flow_tte/gb_cmf.py' \
      '${snapshot}/src/flow_tte/pipeline.py' \
      '${snapshot}/scripts/run_gb_cmf_stage_minus1a.py' \
      '${snapshot}/scripts/aggregate_gb_cmf_stage_minus1a.py'
    test -d '${DATA_ROOT}/can/train/good'; test -d '${DATA_ROOT}/rice/train/good'
    TORCH_HOME=/workspace/.cache/torch FMAD_DINOV3_OFFLINE=1 \
      PYTHONPATH='${snapshot}:${snapshot}/src:${snapshot}/scripts' \
      python3 '${snapshot}/scripts/preflight_dinov3_cache.py' \
        --model dinov3_vith16plus --device cpu --feature-layers 7,15,23,31 \
      >/tmp/${RUN_NAME}_dinov3.log
    mkdir -p '${result_root}/chunks'
    {
      echo 'runner_sha256=${runner_hash}'
      echo 'core_sha256=${core_hash}'
      echo 'pipeline_sha256=${pipeline_hash}'
      echo 'aggregator_sha256=${aggregate_hash}'
      echo 'controller_sha256=${controller_hash}'
      echo 'support_manifest_sha256=${SUPPORT_SHA256}'
      echo \"free_bytes=\${free_bytes}\"
      echo 'container_cuda_count=4'
    } >'${result_root}/preflight.txt'
  '"
}

start_remote() {
  require_run_name; assert_container
  local archive hash
  archive="$(mktemp "${TMPDIR:-/tmp}/gb-cmf-source.XXXXXX.tar")"
  trap 'rm -f -- "'"${archive}"'"' EXIT
  make_archive "${archive}"
  hash="$(sha256sum "${archive}" | awk '{print $1}')"
  stream_snapshot "${archive}" "${hash}"
  preflight
  gpu_idle_check
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    snapshot=${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo
    result_root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    nohup env RUN_NAME=${RUN_NAME} REMOTE_SNAPSHOT_ROOT=${REMOTE_SNAPSHOT_ROOT} \
      REMOTE_RESULTS_ROOT=${REMOTE_RESULTS_ROOT} DATA_ROOT=${DATA_ROOT} \
      TORCH_HOME=/workspace/.cache/torch FMAD_DINOV3_OFFLINE=1 \
      bash \"\${snapshot}/scripts/run_gb_cmf_stage_minus1a_remote.sh\" internal-controller \
      >\"\${result_root}/controller.log\" 2>&1 </dev/null &
    printf '%s\\n' \"\$!\" >\"\${result_root}/controller.pid\"
    printf 'STARTED run_name=%s pid=%s\\n' '${RUN_NAME}' \"\$!\"
  '"
  rm -f "${archive}"; trap - EXIT
}

run_case() {
  local gpu="$1" object_name="$2" seed="$3"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local chunk="${root}/chunks/gpu${gpu}_${object_name}_seed${seed}"
  test ! -e "${chunk}"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${snapshot}:${snapshot}/src:${snapshot}/scripts" \
    python3 "${snapshot}/scripts/run_gb_cmf_stage_minus1a.py" \
      --data-root "${DATA_ROOT}" --support-json "${snapshot}/${SUPPORT_REL}" \
      --output-root "${chunk}" --object "${object_name}" --device cuda \
      --host-gpu-id "${gpu}" --container-cuda-slot 0 --experiment-seed "${seed}" \
      >"${root}/chunks/gpu${gpu}_${object_name}_seed${seed}.log" 2>&1
  printf 'gpu=%s object=%s seed=%s status=complete\n' "${gpu}" "${object_name}" "${seed}" \
    >"${root}/gpu${gpu}_complete.txt"
}

internal_controller() {
  require_run_name
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  trap 'code=$?; if ((code!=0)) && [[ ! -f "'"${root}"'/failed.txt" ]]; then printf "exit_code=%s\n" "${code}" >"'"${root}"'/failed.txt"; fi' EXIT
  exec 9>"${REMOTE_SNAPSHOT_ROOT}/.gb_cmf_stage_minus1a_gpus_0_3.lock"
  flock -n 9 || { echo cooperative_lock_busy >"${root}/failed.txt"; return 75; }
  printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" >"${root}/started.txt"
  run_case 0 rice 0 & p0=$!; run_case 1 can 1 & p1=$!
  run_case 2 rice 1 & p2=$!; run_case 3 can 0 & p3=$!
  status=0; for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do wait "${pid}" || status=1; done
  ((status == 0)) || { echo shard_failure >"${root}/failed.txt"; return 1; }
  test "$(find "${root}/chunks" -name summary.json | wc -l)" -eq 4
  PYTHONPATH="${snapshot}:${snapshot}/src" python3 \
    "${snapshot}/scripts/aggregate_gb_cmf_stage_minus1a.py" --run-root "${root}" \
    >"${root}/aggregate.log" 2>&1
  if find "${root}" -type d \( -name anomaly_maps -o -name score_maps -o \
      -name dense_feature_cache -o -name transformed_images -o -name patch_score_cache \) \
      -print -quit | grep -q .; then echo dense_artifact_found >"${root}/failed.txt"; return 1; fi
  (cd "${root}" && sha256sum aggregate.log summary.json verdict.txt \
    chunks/*/summary.json chunks/*/arms/*/result.json chunks/*/arms/*/complete.txt \
    chunks/*/arms/*/result.sha256 \
    >compact_artifact_sha256.txt)
  printf 'status=complete\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" >"${root}/complete.txt"
}

status_remote() {
  require_run_name; assert_container
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    test -f \"\${root}/complete.txt\" && status=complete || \
      { test -f \"\${root}/failed.txt\" && status=failed || status=running_or_pending; }
    echo 'run_name=${RUN_NAME}'
    echo \"status=\${status}\"
    echo \"gpu_cases_complete=\$(find \"\${root}\" -maxdepth 1 -name 'gpu*_complete.txt' 2>/dev/null | wc -l)\"
    echo \"case_summaries=\$(find \"\${root}/chunks\" -name summary.json 2>/dev/null | wc -l)\"
    echo \"arm_artifacts=\$(find \"\${root}/chunks\" -path '*/arms/*/complete.txt' 2>/dev/null | wc -l)\"
    test -f \"\${root}/failed.txt\" && cat \"\${root}/failed.txt\" || true
  '"
  ssh_remote "${REMOTE_HOST}" \
    "nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
}

pull_remote() {
  require_run_name; assert_container
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}"
  local partial="${LOCAL_PULL_ROOT}/${RUN_NAME}.partial.$$"
  [[ ! -e "${destination}" ]] || die "destination exists: ${destination}"
  mkdir -p "${partial}"
  trap 'rm -rf -- "'"${partial}"'"' EXIT
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' tar -C \
    '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' -cf - ." | tar -C "${partial}" -xf -
  test -f "${partial}/complete.txt"; (cd "${partial}" && sha256sum -c compact_artifact_sha256.txt)
  mv "${partial}" "${destination}"; trap - EXIT
  printf 'PULLED %s\n' "${destination}"
}

cleanup_remote() {
  require_run_name; assert_container
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}"
  [[ "${snapshot}" == /workspace/isolated_runs/gb_cmf_stage_minus1a_* ]] || die "unsafe cleanup"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    test -f '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/complete.txt'
    rm -rf -- '${snapshot}'
    test ! -e '${snapshot}'
  '"
}

invalidate_remote() {
  require_run_name; assert_container
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    controller_pid=\$(tr -cd 0-9 <\"\${root}/controller.pid\")
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
    for _ in \$(seq 1 30); do
      kill -0 \"\${controller_pid}\" 2>/dev/null || break
      sleep 1
    done
    kill -0 \"\${controller_pid}\" 2>/dev/null && exit 76
    printf 'invalidated_before_metrics=post_launch_protocol_audit\n' >\"\${root}/invalidated.txt\"
  '"
}

pull_invalid_remote() {
  require_run_name; assert_container
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}_invalid"
  [[ ! -e "${destination}" ]] || die "destination exists: ${destination}"
  mkdir -p "${destination}"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' tar -C \
    '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' -cf - ." | tar -C "${destination}" -xf -
  test -f "${destination}/invalidated.txt"
  printf 'PULLED_INVALID %s\n' "${destination}"
}

cleanup_invalid_remote() {
  require_run_name; assert_container
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}"
  [[ "${snapshot}" == /workspace/isolated_runs/gb_cmf_stage_minus1a_* ]] || die "unsafe cleanup"
  [[ -f "${LOCAL_PULL_ROOT}/${RUN_NAME}_invalid/invalidated.txt" || \
     -f "${LOCAL_PULL_ROOT}/${RUN_NAME}_invalid/failed.txt" ]] || die "invalid evidence not pulled"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    test -f '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/invalidated.txt' || \
      test -f '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/failed.txt'
    rm -rf -- '${snapshot}' '${REMOTE_RESULTS_ROOT}/${RUN_NAME}'
    test ! -e '${snapshot}'; test ! -e '${REMOTE_RESULTS_ROOT}/${RUN_NAME}'
  '"
}

case "${MODE}" in
  start) RUN_NAME="${RUN_NAME:-gb_cmf_stage_minus1a_$(date -u +%Y%m%dT%H%M%SZ)_$$}"; load_preset; start_remote ;;
  status) load_preset; status_remote ;;
  pull) load_preset; pull_remote ;;
  cleanup) load_preset; cleanup_remote ;;
  invalidate) load_preset; invalidate_remote ;;
  pull-invalid) load_preset; pull_invalid_remote ;;
  cleanup-invalid) load_preset; cleanup_invalid_remote ;;
  internal-controller) internal_controller ;;
  *) die "usage: RUN_NAME=gb_cmf_stage_minus1a_<id> $0 {start|status|pull|cleanup|invalidate|pull-invalid|cleanup-invalid}" ;;
esac
