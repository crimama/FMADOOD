#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${ROOT_DIR}/configs/remote/dsba3.env"
ASKPASS_SCRIPT="${ROOT_DIR}/scripts/ssh_askpass_env.sh"
FIXED_CONTAINER="hun_fsad_tta_012"
REMOTE_SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT:-/var/tmp/fsad_tta/isolated_runs}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/var/tmp/fsad_tta/results_remote}"
REMOTE_PYTEST_ROOT="${REMOTE_PYTEST_ROOT:-/var/tmp/fsad_tta/test_tools}"
LOCAL_PULL_ROOT="${LOCAL_PULL_ROOT:-${ROOT_DIR}/results/remote_runs/dsba3}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-same16_baseline_metric_contract_20260723_v6}"
SUPPORT_REL="skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json"
SUPPORT_SHA256="7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3"
PREREG_REL="skill_graph/experiments/2026-07-24_same16_dvt_density_factorial/preregistration.md"
AGGREGATOR_REL="scripts/aggregate_same16_dvt_density_factorial.py"
CONTROLLER_REL="scripts/run_same16_dvt_density_factorial_remote.sh"
MIN_FREE_GIB="${MIN_FREE_GIB:-30}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-512}"
GPU_IDLE_UTILIZATION="${GPU_IDLE_UTILIZATION:-5}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_run_name() {
  [[ -n "${RUN_NAME:-}" ]] || die "RUN_NAME is required for ${MODE}"
  [[ "${RUN_NAME}" =~ ^same16_dvt_density_factorial_[A-Za-z0-9._-]+$ ]] \
    || die "RUN_NAME must use the same16_dvt_density_factorial_ prefix"
}

load_preset() {
  [[ -f "${PRESET}" ]] || die "missing dsba3 preset: ${PRESET}"
  # shellcheck disable=SC1090
  source "${PRESET}"
  [[ -n "${REMOTE_HOST:-}" && -n "${REMOTE_PORT:-}" ]] \
    || die "remote preset is missing REMOTE_HOST or REMOTE_PORT"
  [[ "${CONTAINER_NAME:-${FIXED_CONTAINER}}" == "${FIXED_CONTAINER}" ]] \
    || die "factorial diagnostic must use fixed container ${FIXED_CONTAINER}"
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

host_gpu_idle_check() {
  ssh_remote "${REMOTE_HOST}" bash -s -- \
    "${GPU_IDLE_MEMORY_MIB}" "${GPU_IDLE_UTILIZATION}" <<'REMOTE_GPU_CHECK'
set -euo pipefail
memory_limit="$1"; util_limit="$2"
for gpu in 0 1 2 3; do
  sample="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits)"
  memory="$(cut -d, -f1 <<<"${sample}" | tr -d '[:space:]')"
  utilization="$(cut -d, -f2 <<<"${sample}" | tr -d '[:space:]')"
  [[ "${memory}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]]
  ((memory <= memory_limit && utilization <= util_limit))
  processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid \
    --format=csv,noheader,nounits)"
  ! grep -Eq '[0-9]+' <<<"${processes}"
  printf 'GPU_IDLE gpu=%s memory_mib=%s utilization=%s compute_apps=0\n' \
    "${gpu}" "${memory}" "${utilization}"
done
REMOTE_GPU_CHECK
}

assert_local_sources() {
  local path
  for path in \
    "${ROOT_DIR}/${SUPPORT_REL}" \
    "${ROOT_DIR}/${PREREG_REL}" \
    "${ROOT_DIR}/${AGGREGATOR_REL}" \
    "${ROOT_DIR}/${CONTROLLER_REL}" \
    "${ROOT_DIR}/tests/test_aggregate_same16_dvt_density_factorial.py" \
    "${ROOT_DIR}/tests/test_run_same16_dvt_density_factorial_remote.py"; do
    [[ -f "${path}" ]] || die "missing launch input: ${path}"
  done
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

stream_snapshot() {
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
  local baseline_root="${REMOTE_RESULTS_ROOT}/${BASELINE_RUN_NAME}"
  ssh_remote "${REMOTE_HOST}" docker exec -i "${FIXED_CONTAINER}" bash -s -- \
    "${snapshot}" "${result_root}" "${baseline_root}" "${DATA_ROOT}" \
    "${SUPPORT_REL}" "${SUPPORT_SHA256}" "$((MIN_FREE_GIB * 1024 * 1024 * 1024))" \
    "${RUN_NAME}" "${REMOTE_PYTEST_ROOT}" <<'REMOTE_PREFLIGHT'
set -euo pipefail
snapshot="$1"; result_root="$2"; baseline_root="$3"; data_root="$4"
support_rel="$5"; support_hash="$6"; min_free_bytes="$7"; run_name="$8"; pytest_root="$9"
test ! -e "${result_root}"
test -f "${baseline_root}/complete.txt"
test -f "${baseline_root}/basic/objects/can/metrics.json"
test -f "${baseline_root}/mechanism/arms/a6/objects/can/metrics.json"
test -f "${baseline_root}/summary.json"
test -f "${baseline_root}/a6_fusion_summary.json"
test "$(sha256sum "${snapshot}/${support_rel}" | awk '{print $1}')" = "${support_hash}"
test "$(python3 -c 'import torch; print(torch.cuda.device_count())')" -eq 4
python3 -c 'import cv2, faiss, numpy, scipy, sklearn, tifffile, torch'
result_parent="$(dirname "${result_root}")"
mkdir -p "${result_parent}"
free_bytes="$(df -PB1 "${result_parent}" | awk 'NR == 2 {print $4}')"
test "${free_bytes}" -ge "${min_free_bytes}"
for object_name in can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts; do
  for split in train/good test_public/good test_public/bad; do
    test -n "$(find "${data_root}/${object_name}/${split}" -maxdepth 1 \
      -type f -name '*.png' -print -quit)"
  done
done
TORCH_HOME=/workspace/.cache/torch FMAD_DINOV2_OFFLINE=1 \
  PYTHONPATH="${snapshot}" python3 "${snapshot}/scripts/preflight_dinov2_cache.py" \
    --model dinov2_vitl14 >"/tmp/${run_name}_dinov2_preflight.log"
cd "${snapshot}"
python3 -m py_compile \
  scripts/aggregate_same16_dvt_density_factorial.py \
  scripts/run_flow_tte_mvtec_ad2.py \
  scripts/flow_tte_mvtec_ad2_core.py
bash -n scripts/run_same16_dvt_density_factorial_remote.sh
PYTHONPATH="${pytest_root}" python3 -m pytest -q \
  tests/test_aggregate_same16_dvt_density_factorial.py \
  tests/test_run_same16_dvt_density_factorial_remote.py \
  tests/test_post_eval_numeric_stability.py \
  tests/test_flow_tte_postprocess_eval.py
printf 'PREFLIGHT_OK run=%s baseline=%s free_bytes=%s\n' \
  "${run_name}" "${baseline_root}" "${free_bytes}"
REMOTE_PREFLIGHT
}

start_controller() {
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  ssh_remote "${REMOTE_HOST}" docker exec -d \
    -e RUN_NAME="${RUN_NAME}" \
    -e BASELINE_RUN_NAME="${BASELINE_RUN_NAME}" \
    -e REMOTE_SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT}" \
    -e REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT}" \
    -e DATA_ROOT="${DATA_ROOT}" \
    -e TORCH_HOME=/workspace/.cache/torch \
    -e FMAD_DINOV2_OFFLINE=1 \
    "${FIXED_CONTAINER}" \
    bash "${snapshot}/${CONTROLLER_REL}" internal-controller
}

run_factorial_object() {
  local snapshot="$1" result_root="$2" gpu="$3" object_name="$4" arm="$5"
  local dvt density
  case "${arm}" in
    dvt_only)
      dvt="position_mean"; density="0.0"
      ;;
    density_only)
      dvt="none"; density="0.25"
      ;;
    *)
      die "unknown factorial arm: ${arm}"
      ;;
  esac
  local output="${result_root}/mechanism/arms/${arm}/objects/${object_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${snapshot}" \
    python3 "${snapshot}/scripts/run_flow_tte_mvtec_ad2.py" \
      --data-root "${DATA_ROOT}" --project-root "${snapshot}" --fsad-root "${snapshot}" \
      --output-root "${output}" --objects "${object_name}" \
      --shots 16 --seed 0 --device cuda --flow-epochs 3 \
      --coupling-layers 2 --hidden-multiplier 1 --flow-lr 2e-4 --flow-clamp 1.9 \
      --flow-transform-mode flow --tail-weight 0.3 --tail-top-k-ratio 0.05 \
      --lambda-logdet 1e-3 --density-quantile 0.90 --expansion-budget 1.0 \
      --distance-weight 1.0 --density-weight "${density}" --score-mode latent_distance \
      --top-percent 0.01 --query-chunk-size 512 --calibration-sample-size 0 \
      --loo-standardization off --latent-bank-subsample none \
      --pro-integration-limit 0.05 \
      --rgb-guide none --threshold-calibration-mode none --binary-postprocess none \
      --backbone-model dinov2_vitl14 --feature-layers 5,11,17,23 \
      --support-selection "fixed_json=${snapshot}/${SUPPORT_REL}" \
      --support-selection-seed 0 --support-transforms identity \
      --feature-fusion layer_norm_mean --normality-mode fused \
      --dvt-denoise-mode "${dvt}" --dvt-denoise-alpha 1.0 \
      --context-source none --flow-context-source auto --memory-context-source auto \
      --context-mode none --context-weight 0.0 --context-top-m 1 \
      --flow-condition-mode none --transformer-context-mode none
}

run_group() {
  local snapshot="$1" result_root="$2" gpu="$3"
  shift 3
  local object_name arm
  for object_name in "$@"; do
    mkdir -p "${result_root}/logs/gpu${gpu}"
    for arm in dvt_only density_only; do
      run_factorial_object "${snapshot}" "${result_root}" "${gpu}" "${object_name}" "${arm}" \
        >"${result_root}/logs/gpu${gpu}/${object_name}_${arm}.log" 2>&1
    done
  done
}

internal_controller() {
  require_run_name
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local baseline_root="${REMOTE_RESULTS_ROOT}/${BASELINE_RUN_NAME}"
  test -d "${snapshot}"
  test ! -e "${result_root}"
  test -f "${baseline_root}/complete.txt"
  mkdir -p "${result_root}/logs"
  trap 'status=$?; if ((status != 0)) && [[ ! -f "${result_root}/failed.txt" ]]; then \
    printf "FAILED controller_exit=%s\n" "${status}" >"${result_root}/failed.txt"; fi' EXIT
  exec > >(tee -a "${result_root}/controller.log") 2>&1
  printf '%s\n' "$$" >"${result_root}/controller.pid"
  cp "${snapshot}/snapshot_hash.txt" "${result_root}/snapshot_hash.txt"
  printf 'baseline_root=%s\n' "${baseline_root}" >"${result_root}/baseline_source.txt"
  cd "${snapshot}"
  local pids=()
  run_group "${snapshot}" "${result_root}" 0 can rice & pids+=("$!")
  run_group "${snapshot}" "${result_root}" 1 fabric walnuts & pids+=("$!")
  run_group "${snapshot}" "${result_root}" 2 vial sheet_metal & pids+=("$!")
  run_group "${snapshot}" "${result_root}" 3 fruit_jelly wallplugs & pids+=("$!")
  local failed=0 pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if ((failed)); then
    printf 'FAILED object_run\n' >"${result_root}/failed.txt"
    exit 1
  fi
  if ! PYTHONPATH="${snapshot}" python3 "${snapshot}/${AGGREGATOR_REL}" \
    --factorial-root "${result_root}" \
    --baseline-root "${baseline_root}" \
    --support-manifest "${snapshot}/${SUPPORT_REL}" \
    >"${result_root}/aggregate.log" 2>&1; then
    printf 'FAILED factorial_aggregate\n' >"${result_root}/failed.txt"
    exit 1
  fi
  while IFS= read -r maps_dir; do
    find "${maps_dir}" -type f -delete
    find "${maps_dir}" -depth -type d -empty -delete
  done < <(
    find "${result_root}" -depth -type d \
      \( -name anomaly_maps -o -name calibration_maps \) -print
  )
  if find "${result_root}" -type d \
    \( -name anomaly_maps -o -name calibration_maps \) -print -quit | grep -q .; then
    printf 'FAILED dense_maps_remain\n' >"${result_root}/failed.txt"
    exit 1
  fi
  printf 'dense_anomaly_and_calibration_map_directories=0\n' \
    >"${result_root}/cleanup_evidence.txt"
  printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >"${result_root}/complete.txt"
  cd /workspace
  find "${snapshot}" -type f -delete
  find "${snapshot}" -depth -type d -empty -delete
  printf 'source_snapshot_removed=true\n' >>"${result_root}/cleanup_evidence.txt"
  trap - EXIT
}

start() {
  require_run_name
  load_preset
  command -v ssh >/dev/null || die "ssh is required"
  assert_local_sources
  ssh_remote "${REMOTE_HOST}" \
    "test \"\$(docker inspect -f '{{.State.Running}}' '${FIXED_CONTAINER}')\" = true"
  host_gpu_idle_check
  local tmp_dir archive archive_hash
  tmp_dir="$(mktemp -d)"
  archive="${tmp_dir}/source.tar"
  make_snapshot_archive "${archive}"
  archive_hash="$(sha256sum "${archive}" | awk '{print $1}')"
  stream_snapshot "${archive}" "${archive_hash}"
  find "${tmp_dir}" -type f -delete
  find "${tmp_dir}" -depth -type d -empty -delete
  container_preflight
  start_controller
  printf 'STARTED run=%s baseline=%s\n' "${RUN_NAME}" "${BASELINE_RUN_NAME}"
}

status() {
  require_run_name
  load_preset
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  ssh_remote "${REMOTE_HOST}" docker exec -i "${FIXED_CONTAINER}" bash -s -- \
    "${result_root}" <<'REMOTE_STATUS'
set -euo pipefail
root="$1"
if [[ -f "${root}/complete.txt" ]]; then
  cat "${root}/complete.txt"
  cat "${root}/dvt_density_factorial_summary.tsv"
elif [[ -f "${root}/failed.txt" ]]; then
  cat "${root}/failed.txt"
  test -f "${root}/aggregate.log" && tail -n 80 "${root}/aggregate.log"
  exit 1
elif [[ -f "${root}/controller.pid" ]]; then
  pid="$(cat "${root}/controller.pid")"
  printf 'RUNNING pid=%s alive=%s metrics=%s\n' \
    "${pid}" \
    "$(kill -0 "${pid}" 2>/dev/null && echo true || echo false)" \
    "$(find "${root}" -name metrics.json -type f | wc -l)"
else
  printf 'NOT_FOUND\n'
fi
REMOTE_STATUS
}

pull() {
  require_run_name
  load_preset
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}"
  [[ ! -e "${destination}" ]] || die "local pull destination exists: ${destination}"
  mkdir -p "${LOCAL_PULL_ROOT}"
  ssh_remote "${REMOTE_HOST}" docker exec "${FIXED_CONTAINER}" \
    test -f "${REMOTE_RESULTS_ROOT}/${RUN_NAME}/complete.txt"
  ssh_remote "${REMOTE_HOST}" docker exec "${FIXED_CONTAINER}" \
    tar -C "${REMOTE_RESULTS_ROOT}" -cf - "${RUN_NAME}" \
    | tar -C "${LOCAL_PULL_ROOT}" -xf -
  test -f "${destination}/dvt_density_factorial_summary.json"
  test -f "${destination}/dvt_density_factorial_summary.tsv"
  test -f "${destination}/cleanup_evidence.txt"
  test -z "$(find "${destination}" -type d \
    \( -name anomaly_maps -o -name calibration_maps \) -print -quit)"
  printf 'PULLED_DIAGNOSTIC destination=%s\n' "${destination}"
}

case "${MODE}" in
  start) start ;;
  status) status ;;
  pull) pull ;;
  internal-controller) internal_controller ;;
  *) die "usage: $0 {start|status|pull|internal-controller}" ;;
esac
