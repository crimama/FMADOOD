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
LOCAL_PULL_ROOT="${LOCAL_PULL_ROOT:-${ROOT_DIR}/results/remote_runs/dsba3}"
SUPPORT_REL="skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json"
SUPPORT_SHA256="7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3"
PROTOCOL_REL="skill_graph/analysis/2026-07-19_dinov2l_default_mechanism_experiment_plan.md"
RUNNER_REL="scripts/run_dinov2l_default_mechanism_stage_m1.py"
AGGREGATOR_REL="scripts/aggregate_dinov2l_default_mechanism_stage_m1.py"
CONTROLLER_REL="scripts/run_dinov2l_default_mechanism_stage_m1_remote.sh"
MIN_FREE_GIB="${MIN_FREE_GIB:-50}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-512}"
GPU_IDLE_UTILIZATION="${GPU_IDLE_UTILIZATION:-5}"

# Keep every object's arms and folds on one device.  This mapping is part of
# the frozen protocol and must not be load-balanced from observed runtimes.
GROUP_0="can,rice"
GROUP_1="fabric,walnuts"
GROUP_2="vial,sheet_metal"
GROUP_3="fruit_jelly,wallplugs"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_run_name() {
  [[ -n "${RUN_NAME:-}" ]] || die "RUN_NAME is required for ${MODE}"
  [[ "${RUN_NAME}" =~ ^dinov2l_default_mechanism_stage_m1_[A-Za-z0-9._-]+$ ]] \
    || die "RUN_NAME must start with dinov2l_default_mechanism_stage_m1_ and use safe characters"
}

load_preset() {
  [[ -f "${PRESET}" ]] || die "missing dsba3 preset: ${PRESET}"
  # shellcheck disable=SC1090
  source "${PRESET}"
  [[ -n "${REMOTE_HOST:-}" && -n "${REMOTE_PORT:-}" ]] \
    || die "remote preset is missing REMOTE_HOST or REMOTE_PORT"
  [[ "${CONTAINER_NAME:-${FIXED_CONTAINER}}" == "${FIXED_CONTAINER}" ]] \
    || die "mechanism audit must use fixed container ${FIXED_CONTAINER}"
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
  # This check is read-only and never waits for, signals, or preempts work.
  ssh_remote "${REMOTE_HOST}" bash -s -- \
    "${GPU_IDLE_MEMORY_MIB}" "${GPU_IDLE_UTILIZATION}" <<'REMOTE_GPU_CHECK'
set -euo pipefail
memory_limit="$1"
util_limit="$2"
for gpu in 0 1 2 3; do
  sample="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits)"
  memory="$(cut -d, -f1 <<<"${sample}" | tr -d '[:space:]')"
  utilization="$(cut -d, -f2 <<<"${sample}" | tr -d '[:space:]')"
  [[ "${memory}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]]
  if ((memory > memory_limit || utilization > util_limit)); then
    printf 'GPU_BUSY gpu=%s memory_mib=%s utilization=%s\n' \
      "${gpu}" "${memory}" "${utilization}" >&2
    exit 73
  fi
  processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid \
    --format=csv,noheader,nounits)"
  if grep -Eq '[0-9]+' <<<"${processes}"; then
    printf 'GPU_COMPUTE_PROCESS_PRESENT gpu=%s pids=%s\n' \
      "${gpu}" "$(tr '\n' ',' <<<"${processes}")" >&2
    exit 74
  fi
  printf 'GPU_IDLE gpu=%s memory_mib=%s utilization=%s compute_apps=0\n' \
    "${gpu}" "${memory}" "${utilization}"
done
REMOTE_GPU_CHECK
}

assert_local_sources() {
  local path
  for path in \
    "${ROOT_DIR}/${RUNNER_REL}" \
    "${ROOT_DIR}/${AGGREGATOR_REL}" \
    "${ROOT_DIR}/${CONTROLLER_REL}" \
    "${ROOT_DIR}/${PROTOCOL_REL}" \
    "${ROOT_DIR}/${SUPPORT_REL}"; do
    [[ -f "${path}" ]] || die "missing launch input: ${path}"
  done
  git -C "${ROOT_DIR}" ls-files --error-unmatch "${SUPPORT_REL}" >/dev/null 2>&1 \
    || die "support manifest is not project-tracked: ${SUPPORT_REL}"
  [[ "$(sha256sum "${ROOT_DIR}/${SUPPORT_REL}" | awk '{print $1}')" == "${SUPPORT_SHA256}" ]] \
    || die "local support manifest SHA-256 mismatch"
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
  local runner_hash aggregator_hash controller_hash protocol_hash
  runner_hash="$(sha256sum "${ROOT_DIR}/${RUNNER_REL}" | awk '{print $1}')"
  aggregator_hash="$(sha256sum "${ROOT_DIR}/${AGGREGATOR_REL}" | awk '{print $1}')"
  controller_hash="$(sha256sum "${ROOT_DIR}/${CONTROLLER_REL}" | awk '{print $1}')"
  protocol_hash="$(sha256sum "${ROOT_DIR}/${PROTOCOL_REL}" | awk '{print $1}')"

  ssh_remote "${REMOTE_HOST}" docker exec -i "${FIXED_CONTAINER}" bash -s -- \
    "${snapshot}" "${result_root}" "${DATA_ROOT}" "${SUPPORT_REL}" \
    "${SUPPORT_SHA256}" "${RUNNER_REL}" "${runner_hash}" \
    "${AGGREGATOR_REL}" "${aggregator_hash}" "${CONTROLLER_REL}" \
    "${controller_hash}" "${PROTOCOL_REL}" "${protocol_hash}" \
    "$((MIN_FREE_GIB * 1024 * 1024 * 1024))" "${RUN_NAME}" <<'REMOTE_PREFLIGHT'
set -euo pipefail
snapshot="$1"; result_root="$2"; data_root="$3"; support_rel="$4"
support_hash="$5"; runner_rel="$6"; runner_hash="$7"
aggregator_rel="$8"; aggregator_hash="$9"; shift 9
controller_rel="$1"; controller_hash="$2"; protocol_rel="$3"; protocol_hash="$4"
min_free_bytes="$5"; run_name="$6"

test ! -e "${result_root}"
command -v flock >/dev/null
command -v sha256sum >/dev/null
command -v nvidia-smi >/dev/null
python3 -c 'import cv2, numpy, sklearn, torch'
test "$(python3 -c 'import torch; print(torch.cuda.device_count())')" -eq 4
free_bytes="$(df -PB1 /workspace | awk 'NR == 2 {print $4}')"
test "${free_bytes}" -ge "${min_free_bytes}"
test "$(sha256sum "${snapshot}/${support_rel}" | awk '{print $1}')" = "${support_hash}"
test "$(sha256sum "${snapshot}/${runner_rel}" | awk '{print $1}')" = "${runner_hash}"
test "$(sha256sum "${snapshot}/${aggregator_rel}" | awk '{print $1}')" = "${aggregator_hash}"
test "$(sha256sum "${snapshot}/${controller_rel}" | awk '{print $1}')" = "${controller_hash}"
test "$(sha256sum "${snapshot}/${protocol_rel}" | awk '{print $1}')" = "${protocol_hash}"
python3 -m py_compile "${snapshot}/${runner_rel}" "${snapshot}/${aggregator_rel}"

python3 - "${snapshot}/${support_rel}" "${data_root}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
data_root = Path(sys.argv[2])
objects = {
    "can", "fabric", "fruit_jelly", "rice",
    "sheet_metal", "vial", "wallplugs", "walnuts",
}
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if set(manifest) != objects:
    raise SystemExit("support manifest object set mismatch")
all_paths = []
for object_name in sorted(objects):
    paths = manifest[object_name]
    if len(paths) != 16 or len(set(paths)) != 16:
        raise SystemExit(f"{object_name}: expected 16 unique support paths")
    expected_parent = data_root / object_name / "train" / "good"
    if not expected_parent.is_dir():
        raise SystemExit(f"missing train/good directory: {expected_parent}")
    for value in paths:
        path = Path(value)
        if path.parent != expected_parent or not path.is_file():
            raise SystemExit(f"invalid or missing support path: {path}")
    all_paths.extend(paths)
if len(all_paths) != 128 or len(set(all_paths)) != 128:
    raise SystemExit("expected 128 globally unique support paths")
PY

TORCH_HOME=/workspace/.cache/torch FMAD_DINOV2_OFFLINE=1 PYTHONPATH="${snapshot}" \
  python3 "${snapshot}/scripts/preflight_dinov2_cache.py" --model dinov2_vitl14 \
  >"/tmp/${run_name}_dinov2_preflight.log"

mkdir -p "${result_root}"
(
  cd "${snapshot}"
  sha256sum "${runner_rel}" "${aggregator_rel}" "${controller_rel}" \
    "${protocol_rel}" "${support_rel}" >"${result_root}/snapshot_files.sha256"
)
{
  printf 'run_name=%s\n' "${run_name}"
  printf 'snapshot=%s\n' "${snapshot}"
  cat "${snapshot}/snapshot_hash.txt"
  printf 'runner_sha256=%s\naggregator_sha256=%s\ncontroller_sha256=%s\n' \
    "${runner_hash}" "${aggregator_hash}" "${controller_hash}"
  printf 'protocol_sha256=%s\nsupport_manifest_sha256=%s\n' \
    "${protocol_hash}" "${support_hash}"
  printf 'support_object_count=8\nsupport_path_count=128\n'
  printf 'free_bytes=%s\ncontainer_cuda_count=4\n' "${free_bytes}"
  printf 'cublas_workspace_config=:4096:8\n'
  printf 'access_scope=train/good_only\ntta=forbidden\nquery_update=forbidden\n'
} >"${result_root}/preflight.txt"
REMOTE_PREFLIGHT
}

run_remote_smoke() {
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  ssh_remote "${REMOTE_HOST}" docker exec -i "${FIXED_CONTAINER}" bash -s -- \
    "${snapshot}" "${result_root}" "${DATA_ROOT}" "${SUPPORT_REL}" "${RUN_NAME}" <<'REMOTE_SMOKE'
set -euo pipefail
snapshot="$1"; result_root="$2"; data_root="$3"; support_rel="$4"; run_name="$5"
smoke_root="${result_root}/smoke"
record_failure() {
  code="$?"
  if ((code != 0)); then
    printf 'phase=smoke exit_code=%s\n' "${code}" >"${result_root}/failed.txt"
  fi
}
trap record_failure EXIT
test ! -e "${smoke_root}"
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
TORCH_HOME=/workspace/.cache/torch FMAD_DINOV2_OFFLINE=1 \
PYTHONPATH="${snapshot}:${snapshot}/src:${snapshot}/scripts" \
  python3 "${snapshot}/scripts/run_dinov2l_default_mechanism_stage_m1.py" \
    --mode smoke \
    --data-root "${data_root}" \
    --support-json "${snapshot}/${support_rel}" \
    --output-root "${smoke_root}" \
    --objects can \
    --device cpu \
    --host-gpu-id smoke \
    --container-cuda-slot cpu \
    >"${result_root}/smoke.log" 2>&1
test -s "${smoke_root}/frozen_config.json"
test -s "${smoke_root}/access_ledger.json"
test -s "${smoke_root}/run_manifest.json"
test -s "${smoke_root}/objects/can.jsonl"
test -s "${smoke_root}/objects/can_summary.json"
printf 'run_name=%s\nstatus=complete\ncompleted_utc=%s\n' \
  "${run_name}" "$(date -u +%FT%TZ)" >"${result_root}/smoke_complete.txt"
REMOTE_SMOKE
}

assert_no_dense_artifacts() {
  local result_root="$1"
  if find "${result_root}" \
      \( -type d \( -name anomaly_maps -o -name score_maps \
          -o -name dense_feature_cache -o -name transformed_images \
          -o -name patch_score_cache \) \
      -o -type f \( -name '*.npy' -o -name '*.npz' -o -name '*.pt' \
          -o -name '*.pth' -o -name '*.pkl' -o -name '*.pickle' \) \) \
      -print -quit | grep -q .; then
    printf 'forbidden dense or binary artifact detected\n' >"${result_root}/failed.txt"
    return 1
  fi
}

write_compact_checksums() {
  local result_root="$1"
  (
    cd "${result_root}"
    find . -type f \
      ! -name compact_artifact_sha256.txt \
      ! -name complete.txt \
      ! -name controller.log \
      ! -name controller.pid \
      -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum >compact_artifact_sha256.txt
  )
  test -s "${result_root}/compact_artifact_sha256.txt"
}

prepare_snapshot() {
  assert_local_sources
  assert_fixed_container
  local archive archive_hash
  archive="$(mktemp "${TMPDIR:-/tmp}/dinov2l-mechanism-m1-source.XXXXXX.tar")"
  trap 'rm -f -- "'"${archive}"'"' EXIT
  make_snapshot_archive "${archive}"
  archive_hash="$(sha256sum "${archive}" | awk '{print $1}')"
  stream_isolated_snapshot "${archive}" "${archive_hash}"
  container_preflight
  rm -f -- "${archive}"
  trap - EXIT
}

smoke_remote() {
  require_run_name
  prepare_snapshot
  run_remote_smoke
  ssh_remote "${REMOTE_HOST}" docker exec -i "${FIXED_CONTAINER}" bash -s -- \
    "${REMOTE_RESULTS_ROOT}/${RUN_NAME}" "${RUN_NAME}" <<'REMOTE_SMOKE_FINALIZE'
set -euo pipefail
result_root="$1"; run_name="$2"
if find "${result_root}" \
    \( -type d \( -name anomaly_maps -o -name score_maps -o -name dense_feature_cache \
        -o -name transformed_images -o -name patch_score_cache \) \
    -o -type f \( -name '*.npy' -o -name '*.npz' -o -name '*.pt' \
        -o -name '*.pth' -o -name '*.pkl' -o -name '*.pickle' \) \) \
    -print -quit | grep -q .; then
  echo forbidden_dense_artifact >"${result_root}/failed.txt"
  exit 1
fi
(
  cd "${result_root}"
  find . -type f ! -name compact_artifact_sha256.txt ! -name complete.txt \
    -print0 | LC_ALL=C sort -z | xargs -0 sha256sum >compact_artifact_sha256.txt
)
printf 'run_name=%s\nrun_type=smoke\nstatus=complete\ncompleted_utc=%s\n' \
  "${run_name}" "$(date -u +%FT%TZ)" >"${result_root}/complete.txt"
REMOTE_SMOKE_FINALIZE
  printf 'SMOKE_COMPLETE run_name=%s\n' "${RUN_NAME}"
}

start_remote() {
  require_run_name
  # Check before staging, then again immediately before the controller starts.
  assert_fixed_container
  host_gpu_idle_check
  prepare_snapshot
  run_remote_smoke
  host_gpu_idle_check
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  ssh_remote "${REMOTE_HOST}" docker exec -i "${FIXED_CONTAINER}" bash -s -- \
    "${snapshot}" "${result_root}" "${RUN_NAME}" "${REMOTE_SNAPSHOT_ROOT}" \
    "${REMOTE_RESULTS_ROOT}" "${DATA_ROOT}" <<'REMOTE_START'
set -euo pipefail
snapshot="$1"; result_root="$2"; run_name="$3"; snapshot_root="$4"
results_root="$5"; data_root="$6"
nohup env \
  RUN_NAME="${run_name}" \
  REMOTE_SNAPSHOT_ROOT="${snapshot_root}" \
  REMOTE_RESULTS_ROOT="${results_root}" \
  DATA_ROOT="${data_root}" \
  TORCH_HOME=/workspace/.cache/torch \
  FMAD_DINOV2_OFFLINE=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  bash "${snapshot}/scripts/run_dinov2l_default_mechanism_stage_m1_remote.sh" \
    internal-controller >"${result_root}/controller.log" 2>&1 </dev/null &
printf '%s\n' "$!" >"${result_root}/controller.pid"
printf 'STARTED run_name=%s pid=%s\n' "${run_name}" "$!"
REMOTE_START
}

run_gpu_group() {
  local gpu="$1" objects="$2"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local chunk_root="${result_root}/chunks/gpu${gpu}_${objects//,/_}"
  test ! -e "${chunk_root}"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  CUDA_VISIBLE_DEVICES="${gpu}" \
  TORCH_HOME=/workspace/.cache/torch \
  FMAD_DINOV2_OFFLINE=1 \
  PYTHONPATH="${snapshot}:${snapshot}/src:${snapshot}/scripts" \
    python3 "${snapshot}/${RUNNER_REL}" \
      --mode real \
      --data-root "${DATA_ROOT}" \
      --support-json "${snapshot}/${SUPPORT_REL}" \
      --output-root "${chunk_root}" \
      --objects "${objects}" \
      --device cuda:0 \
      --host-gpu-id "${gpu}" \
      --container-cuda-slot 0 \
      >"${result_root}/chunks/gpu${gpu}_${objects//,/_}.log" 2>&1
  local expected object_name
  IFS=',' read -r -a expected <<<"${objects}"
  for object_name in "${expected[@]}"; do
    test -s "${chunk_root}/objects/${object_name}.jsonl"
    test -s "${chunk_root}/objects/${object_name}_summary.json"
  done
  test -s "${chunk_root}/frozen_config.json"
  test -s "${chunk_root}/access_ledger.json"
  test -s "${chunk_root}/run_manifest.json"
  printf 'gpu=%s objects=%s status=complete\n' "${gpu}" "${objects}" \
    >"${result_root}/gpu${gpu}_complete.txt"
}

internal_controller() {
  require_run_name
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  mkdir -p "${result_root}/chunks"
  record_controller_failure() {
    local exit_code="$?"
    if [[ "${exit_code}" -ne 0 && ! -f "${result_root}/complete.txt" \
        && ! -f "${result_root}/failed.txt" ]]; then
      printf 'phase=controller exit_code=%s\n' "${exit_code}" >"${result_root}/failed.txt"
    fi
  }
  trap record_controller_failure EXIT
  exec 9>"${REMOTE_SNAPSHOT_ROOT}/.dinov2l_default_mechanism_stage_m1_gpus_0_3.lock"
  if ! flock -n 9; then
    printf 'cooperative GPU 0-3 lock is already held\n' >"${result_root}/failed.txt"
    return 75
  fi
  test "$(python3 -c 'import torch; print(torch.cuda.device_count())')" -eq 4
  test -f "${result_root}/smoke_complete.txt"
  printf 'controller_pid=%s\nstarted_utc=%s\n' "$$" "$(date -u +%FT%TZ)" \
    >"${result_root}/started.txt"

  run_gpu_group 0 "${GROUP_0}" & p0=$!
  run_gpu_group 1 "${GROUP_1}" & p1=$!
  run_gpu_group 2 "${GROUP_2}" & p2=$!
  run_gpu_group 3 "${GROUP_3}" & p3=$!

  local status=0 pid
  for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    printf 'one or more GPU groups failed; aggregation was not run\n' \
      >"${result_root}/failed.txt"
    return 1
  fi
  for gpu in 0 1 2 3; do
    test -f "${result_root}/gpu${gpu}_complete.txt"
  done
  test "$(find "${result_root}/chunks" -path '*/objects/*.jsonl' | wc -l)" -eq 8
  test "$(find "${result_root}/chunks" -path '*/objects/*_summary.json' | wc -l)" -eq 8
  test "$(find "${result_root}/chunks" -name access_ledger.json | wc -l)" -eq 4

  PYTHONPATH="${snapshot}:${snapshot}/src:${snapshot}/scripts" \
    python3 "${snapshot}/${AGGREGATOR_REL}" \
      --run-root "${result_root}" --require-all8 \
      >"${result_root}/aggregate.log" 2>&1
  test -s "${result_root}/summary.json"
  test -s "${result_root}/causal_contrasts.tsv"
  test -s "${result_root}/per_object_metrics.tsv"
  test -s "${result_root}/verdict.txt"

  assert_no_dense_artifacts "${result_root}"
  write_compact_checksums "${result_root}"
  printf 'run_name=%s\nrun_type=full\nstatus=complete\ncompleted_utc=%s\n' \
    "${RUN_NAME}" "$(date -u +%FT%TZ)" >"${result_root}/complete.txt"
}

status_remote() {
  require_run_name
  assert_fixed_container
  ssh_remote "${REMOTE_HOST}" docker exec -i "${FIXED_CONTAINER}" bash -s -- \
    "${REMOTE_RESULTS_ROOT}/${RUN_NAME}" "${RUN_NAME}" <<'REMOTE_STATUS'
set -euo pipefail
root="$1"; run_name="$2"
if test -f "${root}/complete.txt"; then
  status=complete
elif test -f "${root}/failed.txt"; then
  status=failed
elif test -d "${root}"; then
  status=running_or_pending
else
  status=not_found
fi
printf 'run_name=%s\nstatus=%s\n' "${run_name}" "${status}"
printf 'smoke_complete=%s\n' "$(test -f "${root}/smoke_complete.txt" && echo 1 || echo 0)"
printf 'gpu_groups_complete=%s\n' \
  "$(find "${root}" -maxdepth 1 -name 'gpu*_complete.txt' 2>/dev/null | wc -l)"
printf 'object_jsonl=%s\n' \
  "$(find "${root}/chunks" -path '*/objects/*.jsonl' 2>/dev/null | wc -l)"
printf 'object_summaries=%s\n' \
  "$(find "${root}/chunks" -path '*/objects/*_summary.json' 2>/dev/null | wc -l)"
test -f "${root}/failed.txt" && sed -n '1,30p' "${root}/failed.txt" || true
if test -s "${root}/controller.pid"; then
  ps -o pid=,stat=,etime=,args= -p "$(cat "${root}/controller.pid")" || true
fi
REMOTE_STATUS
  ssh_remote "${REMOTE_HOST}" \
    "nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
}

pull_remote() {
  require_run_name
  assert_fixed_container
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}"
  local partial="${destination}.partial.$$"
  [[ ! -e "${destination}" ]] || die "local pull destination exists: ${destination}"
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
  test -s "${partial}/compact_artifact_sha256.txt"
  (cd "${partial}" && sha256sum -c compact_artifact_sha256.txt)
  assert_no_dense_artifacts "${partial}"
  mv "${partial}" "${destination}"
  trap - EXIT
  printf 'PULLED run_name=%s destination=%s\n' "${RUN_NAME}" "${destination}"
}

cleanup_remote() {
  require_run_name
  assert_fixed_container
  local snapshot_root="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}"
  [[ "${snapshot_root}" == /workspace/isolated_runs/dinov2l_default_mechanism_stage_m1_* ]] \
    || die "refusing cleanup outside the isolated mechanism-audit namespace"
  ssh_remote "${REMOTE_HOST}" docker exec -i "${FIXED_CONTAINER}" bash -s -- \
    "${REMOTE_RESULTS_ROOT}/${RUN_NAME}" "${snapshot_root}" <<'REMOTE_CLEANUP'
set -euo pipefail
result_root="$1"; snapshot_root="$2"
test -f "${result_root}/complete.txt"
if test -s "${result_root}/controller.pid"; then
  controller_pid="$(cat "${result_root}/controller.pid")"
  if kill -0 "${controller_pid}" 2>/dev/null; then
    controller_state="$(ps -o stat= -p "${controller_pid}" 2>/dev/null \
      | awk 'NR == 1 {print $1}')"
    if [[ -n "${controller_state}" && "${controller_state}" != Z* ]]; then
      echo 'refusing cleanup while controller is active' >&2
      exit 76
    fi
  fi
fi
rm -rf -- "${snapshot_root}"
test ! -e "${snapshot_root}"
printf 'cleanup=complete snapshot_removed=%s compact_results_retained=%s\n' \
  "${snapshot_root}" "${result_root}"
REMOTE_CLEANUP
}

case "${MODE}" in
  smoke)
    RUN_NAME="${RUN_NAME:-dinov2l_default_mechanism_stage_m1_smoke_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
    load_preset
    smoke_remote
    ;;
  start)
    RUN_NAME="${RUN_NAME:-dinov2l_default_mechanism_stage_m1_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
    load_preset
    start_remote
    ;;
  status)
    load_preset
    status_remote
    ;;
  pull)
    load_preset
    pull_remote
    ;;
  cleanup)
    load_preset
    cleanup_remote
    ;;
  internal-controller)
    internal_controller
    ;;
  *)
    die "usage: RUN_NAME=dinov2l_default_mechanism_stage_m1_<id> $0 {smoke|start|status|pull|cleanup}"
    ;;
esac
