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
SUPPORT_REL="skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json"
SUPPORT_SHA256="7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3"
MIN_FREE_GIB="${MIN_FREE_GIB:-50}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-512}"
GPU_IDLE_UTILIZATION="${GPU_IDLE_UTILIZATION:-5}"
LOCAL_PULL_ROOT="${LOCAL_PULL_ROOT:-${ROOT_DIR}/results/remote_runs/dsba3}"

OBJECT_GROUP_0="can fruit_jelly"
OBJECT_GROUP_1="fabric sheet_metal"
OBJECT_GROUP_2="rice wallplugs"
OBJECT_GROUP_3="vial walnuts"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_run_name() {
  [[ -n "${RUN_NAME:-}" ]] || die "RUN_NAME is required for ${MODE}"
  [[ "${RUN_NAME}" =~ ^cfntr_stage_minus1_[A-Za-z0-9._-]+$ ]] \
    || die "RUN_NAME must start with cfntr_stage_minus1_ and contain only safe characters"
}

load_remote_preset() {
  [[ -f "${PRESET}" ]] || die "missing local preset: ${PRESET}"
  # shellcheck disable=SC1090
  source "${PRESET}"
  [[ -n "${REMOTE_HOST:-}" && -n "${REMOTE_PORT:-}" ]] \
    || die "remote preset is missing REMOTE_HOST or REMOTE_PORT"
  [[ "${CONTAINER_NAME:-${FIXED_CONTAINER}}" == "${FIXED_CONTAINER}" ]] \
    || die "CF-NTR must use the fixed container ${FIXED_CONTAINER}"
  CONTAINER_NAME="${FIXED_CONTAINER}"
}

ssh_remote() {
  local args=(-o StrictHostKeyChecking=no -p "${REMOTE_PORT}")
  if [[ -n "${JUMP_HOST:-}" ]]; then
    args=(-J "${JUMP_HOST}:${JUMP_PORT}" "${args[@]}")
  fi
  if [[ -n "${SSH_PASSWORD:-}" ]]; then
    env SSH_PASSWORD="${SSH_PASSWORD}" DISPLAY="${DISPLAY:-dummy}" \
      SSH_ASKPASS="${ASKPASS_SCRIPT}" SSH_ASKPASS_REQUIRE=force \
      setsid -w ssh "${args[@]}" "$@"
  else
    ssh "${args[@]}" "$@"
  fi
}

assert_fixed_container() {
  ssh_remote "${REMOTE_HOST}" \
    "test \"\$(docker inspect -f '{{.State.Running}}' '${FIXED_CONTAINER}' 2>/dev/null)\" = true"
}

host_gpu_idle_check() {
  # This is deliberately a host-side, read-only check.  It never waits for,
  # signals, or preempts another process.
  ssh_remote "${REMOTE_HOST}" bash -s -- \
    "${GPU_IDLE_MEMORY_MIB}" "${GPU_IDLE_UTILIZATION}" <<'REMOTE_GPU_CHECK'
set -euo pipefail
memory_limit="$1"
util_limit="$2"
for gpu in 0 1 2 3; do
  sample="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  memory="$(cut -d, -f1 <<<"${sample}" | tr -d '[:space:]')"
  utilization="$(cut -d, -f2 <<<"${sample}" | tr -d '[:space:]')"
  [[ "${memory}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]]
  if (( memory > memory_limit || utilization > util_limit )); then
    printf 'GPU_BUSY gpu=%s memory_mib=%s utilization=%s\n' "${gpu}" "${memory}" "${utilization}" >&2
    exit 73
  fi
  processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if grep -Eq '[0-9]+' <<<"${processes}"; then
    printf 'GPU_COMPUTE_PROCESS_PRESENT gpu=%s pids=%s\n' "${gpu}" "$(tr '\n' ',' <<<"${processes}")" >&2
    exit 74
  fi
  printf 'GPU_IDLE gpu=%s memory_mib=%s utilization=%s compute_apps=0\n' "${gpu}" "${memory}" "${utilization}"
done
REMOTE_GPU_CHECK
}

make_snapshot_archive() {
  local archive="$1"
  tar \
    --exclude=.git \
    --exclude=.omx \
    --exclude=.venv \
    --exclude=results \
    --exclude=configs/remote \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    -C "${ROOT_DIR}" -cf "${archive}" .
}

stream_isolated_snapshot() {
  local archive="$1" archive_hash="$2"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  ssh_remote "${REMOTE_HOST}" \
    "docker exec -i '${FIXED_CONTAINER}' bash -lc '
      set -euo pipefail
      snapshot=\"${snapshot}\"
      archive=/tmp/${RUN_NAME}_source.tar
      test ! -e \"\${snapshot}\"
      mkdir -p \"\${snapshot}\"
      tee \"\${archive}\" >/dev/null
      actual=\$(sha256sum \"\${archive}\" | awk \"{print \\\$1}\")
      test \"\${actual}\" = \"${archive_hash}\"
      tar -C \"\${snapshot}\" -xf \"\${archive}\"
      rm -f \"\${archive}\"
      printf \"archive_sha256=%s\\n\" \"\${actual}\" >\"\${snapshot}/snapshot_hash.txt\"
    '" <"${archive}"
}

container_preflight() {
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local local_runner_hash local_core_hash local_aggregate_hash
  local_runner_hash="$(sha256sum "${ROOT_DIR}/scripts/run_cfntr_stage_minus1.py" | awk '{print $1}')"
  local_core_hash="$(sha256sum "${ROOT_DIR}/src/flow_tte/cfntr.py" | awk '{print $1}')"
  local_aggregate_hash="$(sha256sum "${ROOT_DIR}/scripts/aggregate_cfntr_stage_minus1.py" | awk '{print $1}')"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    snapshot=\"${snapshot}\"
    result_root=\"${result_root}\"
    test ! -e \"\${result_root}\"
    command -v flock >/dev/null
    command -v sha256sum >/dev/null
    command -v nvidia-smi >/dev/null
    python3 -c \"import cv2, numpy, sklearn, torch\"
    test \"\$(python3 -c \"import torch; print(torch.cuda.device_count())\")\" -eq 4
    free_bytes=\$(df -PB1 /workspace | awk \"NR==2 {print \\\$4}\")
    test \"\${free_bytes}\" -ge $((MIN_FREE_GIB * 1024 * 1024 * 1024))
    test \"\$(sha256sum \"\${snapshot}/${SUPPORT_REL}\" | awk \"{print \\\$1}\")\" = \"${SUPPORT_SHA256}\"
    test \"\$(sha256sum \"\${snapshot}/scripts/run_cfntr_stage_minus1.py\" | awk \"{print \\\$1}\")\" = \"${local_runner_hash}\"
    test \"\$(sha256sum \"\${snapshot}/src/flow_tte/cfntr.py\" | awk \"{print \\\$1}\")\" = \"${local_core_hash}\"
    test \"\$(sha256sum \"\${snapshot}/scripts/aggregate_cfntr_stage_minus1.py\" | awk \"{print \\\$1}\")\" = \"${local_aggregate_hash}\"
    test -d \"${DATA_ROOT}\"
    for object_name in can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts; do
      test -d \"${DATA_ROOT}/\${object_name}/train/good\"
    done
    TORCH_HOME=/workspace/.cache/torch FMAD_DINOV2_OFFLINE=1 PYTHONPATH=\"\${snapshot}\" \\
      python3 \"\${snapshot}/scripts/preflight_dinov2_cache.py\" --model dinov2_vitl14 \\
      >/tmp/${RUN_NAME}_dinov2_preflight.log
    mkdir -p \"\${result_root}\"
    {
      printf \"run_name=%s\\n\" \"${RUN_NAME}\"
      printf \"snapshot=%s\\n\" \"\${snapshot}\"
      cat \"\${snapshot}/snapshot_hash.txt\"
      printf \"runner_sha256=%s\\ncore_sha256=%s\\naggregator_sha256=%s\\n\" \\
        \"${local_runner_hash}\" \"${local_core_hash}\" \"${local_aggregate_hash}\"
      printf \"support_manifest_sha256=%s\\nfree_bytes=%s\\ncontainer_cuda_count=4\\n\" \\
        \"${SUPPORT_SHA256}\" \"\${free_bytes}\"
    } >\"\${result_root}/preflight.txt\"
  '"
}

start_remote() {
  require_run_name
  assert_fixed_container
  local archive archive_hash
  archive="$(mktemp "${TMPDIR:-/tmp}/cfntr-source.XXXXXX.tar")"
  trap 'rm -f -- "'"${archive}"'"' EXIT
  make_snapshot_archive "${archive}"
  archive_hash="$(sha256sum "${archive}" | awk '{print $1}')"
  stream_isolated_snapshot "${archive}" "${archive_hash}"
  container_preflight

  # The second check is intentionally the last host action before launch.
  host_gpu_idle_check
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    snapshot=${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo
    result_root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    nohup env RUN_NAME=${RUN_NAME} REMOTE_SNAPSHOT_ROOT=${REMOTE_SNAPSHOT_ROOT} \\
      REMOTE_RESULTS_ROOT=${REMOTE_RESULTS_ROOT} DATA_ROOT=${DATA_ROOT} \\
      TORCH_HOME=/workspace/.cache/torch FMAD_DINOV2_OFFLINE=1 \\
      bash \"\${snapshot}/scripts/run_cfntr_stage_minus1_remote.sh\" internal-controller \\
      >\"\${result_root}/controller.log\" 2>&1 </dev/null &
    printf \"%s\\n\" \"\$!\" >\"\${result_root}/controller.pid\"
    printf \"STARTED run_name=%s pid=%s snapshot_archive_sha256=%s\\n\" \\
      ${RUN_NAME} \"\$!\" ${archive_hash}
  '"
  rm -f -- "${archive}"
  trap - EXIT
}

run_object() {
  local gpu="$1" object_name="$2"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local chunk_root="${result_root}/chunks/gpu${gpu}_${object_name}"
  mkdir -p "${result_root}/chunks"
  test ! -e "${chunk_root}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${snapshot}:${snapshot}/src" \
    python3 "${snapshot}/scripts/run_cfntr_stage_minus1.py" \
      --data-root "${DATA_ROOT}" \
      --support-json "${snapshot}/${SUPPORT_REL}" \
      --output-root "${chunk_root}" \
      --objects "${object_name}" \
      --device cuda \
      --backbone-model dinov2_vitl14 \
      --host-gpu-id "${gpu}" \
      --container-cuda-slot 0 \
      >"${result_root}/chunks/gpu${gpu}_${object_name}.log" 2>&1
}

run_group() {
  local gpu="$1"
  shift
  local object_name
  for object_name in "$@"; do
    run_object "${gpu}" "${object_name}"
  done
  printf 'gpu=%s status=complete objects=%s\n' "${gpu}" "$*" \
    >"${REMOTE_RESULTS_ROOT}/${RUN_NAME}/gpu${gpu}_complete.txt"
}

internal_controller() {
  require_run_name
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  mkdir -p "${result_root}"
  record_controller_failure() {
    local exit_code="$?"
    if [[ "${exit_code}" -ne 0 && ! -f "${result_root}/complete.txt" \
        && ! -f "${result_root}/failed.txt" ]]; then
      printf 'controller failed with exit_code=%s; inspect controller.log and shard logs\n' \
        "${exit_code}" >"${result_root}/failed.txt"
    fi
  }
  trap record_controller_failure EXIT
  exec 9>"${REMOTE_SNAPSHOT_ROOT}/.cfntr_stage_minus1_gpus_0_3.lock"
  if ! flock -n 9; then
    printf 'cooperative GPU lock is already held\n' >"${result_root}/failed.txt"
    return 75
  fi
  test "$(python3 -c 'import torch; print(torch.cuda.device_count())')" -eq 4
  printf 'controller_pid=%s\nstarted_utc=%s\n' "$$" "$(date -u +%FT%TZ)" \
    >"${result_root}/started.txt"

  read -r -a group0 <<<"${OBJECT_GROUP_0}"
  read -r -a group1 <<<"${OBJECT_GROUP_1}"
  read -r -a group2 <<<"${OBJECT_GROUP_2}"
  read -r -a group3 <<<"${OBJECT_GROUP_3}"
  run_group 0 "${group0[@]}" & p0=$!
  run_group 1 "${group1[@]}" & p1=$!
  run_group 2 "${group2[@]}" & p2=$!
  run_group 3 "${group3[@]}" & p3=$!

  status=0
  for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    printf 'one or more GPU groups failed; aggregation was not run\n' >"${result_root}/failed.txt"
    return "${status}"
  fi
  for gpu in 0 1 2 3; do
    test -f "${result_root}/gpu${gpu}_complete.txt"
  done

  PYTHONPATH="${snapshot}:${snapshot}/src" \
    python3 "${snapshot}/scripts/aggregate_cfntr_stage_minus1.py" \
      --run-root "${result_root}" --require-all8 \
      >"${result_root}/aggregate.log" 2>&1

  if find "${result_root}" -type d \
      \( -name anomaly_maps -o -name score_maps -o -name dense_feature_cache \
         -o -name transformed_images -o -name patch_score_cache \) \
      -print -quit | grep -q .; then
    printf 'non-compact artifact directory detected\n' >"${result_root}/failed.txt"
    return 1
  fi
  (
    cd "${result_root}"
    sha256sum aggregate.log summary.json per_object.tsv bootstrap.tsv verdict.txt \
      chunks/*/*_summary.json >compact_artifact_sha256.txt
  )
  printf 'run_name=%s\nstatus=complete\ncompleted_utc=%s\n' \
    "${RUN_NAME}" "$(date -u +%FT%TZ)" >"${result_root}/complete.txt"
}

status_remote() {
  require_run_name
  assert_fixed_container
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    printf \"run_name=%s\\n\" ${RUN_NAME}
    test -f \"\${root}/complete.txt\" && echo status=complete \\
      || { test -f \"\${root}/failed.txt\" && echo status=failed || echo status=running_or_pending; }
    test -f \"\${root}/failed.txt\" && sed -n \"1,20p\" \"\${root}/failed.txt\" || true
    printf \"gpu_groups_complete=%s\\n\" \"\$(find \"\${root}\" -maxdepth 1 -name \"gpu*_complete.txt\" 2>/dev/null | wc -l)\"
    printf \"object_summaries=%s\\n\" \"\$(find \"\${root}/chunks\" -name \"*_summary.json\" 2>/dev/null | wc -l)\"
    if test -s \"\${root}/controller.pid\"; then
      ps -o pid=,stat=,etime=,args= -p \"\$(cat \"\${root}/controller.pid\")\" || true
    fi
  '"
  ssh_remote "${REMOTE_HOST}" \
    "nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
}

pull_remote() {
  require_run_name
  assert_fixed_container
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}"
  local partial="${destination}.partial.$$"
  [[ ! -e "${destination}" ]] || die "local pull destination already exists: ${destination}"
  mkdir -p "$(dirname "${destination}")" "${partial}"
  trap 'rm -rf -- "'"${partial}"'"' EXIT
  ssh_remote "${REMOTE_HOST}" \
    "docker exec '${FIXED_CONTAINER}' tar \
      --exclude='*/anomaly_maps' --exclude='*/score_maps' \
      --exclude='*/dense_feature_cache' --exclude='*/transformed_images' \
      --exclude='*/patch_score_cache' \
      -C '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' -cf - ." \
    | tar -C "${partial}" -xf -
  test -f "${partial}/complete.txt"
  test -f "${partial}/compact_artifact_sha256.txt"
  (cd "${partial}" && sha256sum -c compact_artifact_sha256.txt)
  if find "${partial}" -type d \
      \( -name anomaly_maps -o -name score_maps -o -name dense_feature_cache \
         -o -name transformed_images -o -name patch_score_cache \) \
      -print -quit | grep -q .; then
    die "compact pull contains a forbidden dense artifact directory"
  fi
  mv "${partial}" "${destination}"
  trap - EXIT
  printf 'PULLED run_name=%s destination=%s\n' "${RUN_NAME}" "${destination}"
}

cleanup_remote() {
  require_run_name
  assert_fixed_container
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}"
  [[ "${snapshot}" == /workspace/isolated_runs/cfntr_stage_minus1_* ]] \
    || die "refusing cleanup outside the isolated CF-NTR snapshot namespace"
  ssh_remote "${REMOTE_HOST}" "docker exec -i '${FIXED_CONTAINER}' bash -s -- \
    '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' '${snapshot}'" <<'REMOTE_CLEANUP'
set -euo pipefail
result_root="$1"
snapshot="$2"
test -f "${result_root}/complete.txt"
if test -s "${result_root}/controller.pid"; then
  controller_pid="$(cat "${result_root}/controller.pid")"
  if kill -0 "${controller_pid}" 2>/dev/null; then
    controller_state="$(ps -o stat= -p "${controller_pid}" 2>/dev/null \
      | awk 'NR == 1 { print $1 }')"
    if [[ -n "${controller_state}" && "${controller_state}" != Z* ]]; then
      echo "refusing cleanup while controller is active" >&2
      exit 76
    fi
  fi
fi
find "${result_root}" -type d \
  \( -name anomaly_maps -o -name score_maps -o -name dense_feature_cache \
     -o -name transformed_images -o -name patch_score_cache \) \
  -prune -exec rm -rf -- {} +
rm -rf -- "${snapshot}"
test ! -e "${snapshot}"
printf 'cleanup=complete snapshot_removed=%s compact_results_retained=%s\n' \
  "${snapshot}" "${result_root}"
REMOTE_CLEANUP
}

case "${MODE}" in
  start)
    if [[ -z "${RUN_NAME:-}" ]]; then
      RUN_NAME="cfntr_stage_minus1_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    fi
    load_remote_preset
    start_remote
    ;;
  status)
    load_remote_preset
    status_remote
    ;;
  pull)
    load_remote_preset
    pull_remote
    ;;
  cleanup)
    load_remote_preset
    cleanup_remote
    ;;
  internal-controller)
    internal_controller
    ;;
  *)
    die "usage: RUN_NAME=cfntr_stage_minus1_<id> $0 {start|status|pull|cleanup}"
    ;;
esac
