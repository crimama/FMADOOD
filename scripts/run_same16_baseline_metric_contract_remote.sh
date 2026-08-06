#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${ROOT_DIR}/configs/remote/dsba3.env"
ASKPASS_SCRIPT="${ROOT_DIR}/scripts/ssh_askpass_env.sh"
FIXED_CONTAINER="hun_fsad_tta_012"
REMOTE_SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT:-/workspace/isolated_runs}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/workspace/results_remote}"
REMOTE_PYTEST_ROOT="${REMOTE_PYTEST_ROOT:-/var/tmp/fsad_tta/test_tools}"
LOCAL_PULL_ROOT="${LOCAL_PULL_ROOT:-${ROOT_DIR}/results/remote_runs/dsba3}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
SUPPORT_REL="skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json"
SUPPORT_SHA256="7e93f15e2efef565354ebe03a9cc5e834464fc4199bd2273067d62eda2ab74b3"
PREREG_REL="skill_graph/experiments/2026-07-23_same16_baseline_metric_contract/preregistration.md"
AGGREGATOR_REL="scripts/aggregate_same16_baselines.py"
A6_PREREG_REL="skill_graph/experiments/2026-07-23_a6_dominant_layer_public_shadow/preregistration.md"
A6_AGGREGATOR_REL="scripts/aggregate_a6_fusion_public_shadow.py"
ADJUDICATOR_REL="scripts/adjudicate_same16_a6.py"
CONTROLLER_REL="scripts/run_same16_baseline_metric_contract_remote.sh"
MIN_FREE_GIB="${MIN_FREE_GIB:-80}"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-512}"
GPU_IDLE_UTILIZATION="${GPU_IDLE_UTILIZATION:-5}"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_run_name() {
  [[ -n "${RUN_NAME:-}" ]] || die "RUN_NAME is required for ${MODE}"
  [[ "${RUN_NAME}" =~ ^same16_baseline_metric_contract_[A-Za-z0-9._-]+$ ]] \
    || die "RUN_NAME must use the same16_baseline_metric_contract_ prefix"
}

load_preset() {
  [[ -f "${PRESET}" ]] || die "missing dsba3 preset: ${PRESET}"
  # shellcheck disable=SC1090
  source "${PRESET}"
  [[ -n "${REMOTE_HOST:-}" && -n "${REMOTE_PORT:-}" ]] \
    || die "remote preset is missing REMOTE_HOST or REMOTE_PORT"
  [[ "${CONTAINER_NAME:-${FIXED_CONTAINER}}" == "${FIXED_CONTAINER}" ]] \
    || die "baseline contract must use fixed container ${FIXED_CONTAINER}"
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
memory_limit="$1"
util_limit="$2"
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
    "${ROOT_DIR}/${A6_PREREG_REL}" \
    "${ROOT_DIR}/${A6_AGGREGATOR_REL}" \
    "${ROOT_DIR}/${ADJUDICATOR_REL}" \
    "${ROOT_DIR}/${CONTROLLER_REL}" \
    "${ROOT_DIR}/configs/baselines/same16_superad_ad2.yaml" \
    "${ROOT_DIR}/configs/baselines/same16_rn_fmlk_ad2.yaml"; do
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
  ssh_remote "${REMOTE_HOST}" docker exec -i "${FIXED_CONTAINER}" bash -s -- \
    "${snapshot}" "${result_root}" "${DATA_ROOT}" "${SUPPORT_REL}" \
    "${SUPPORT_SHA256}" "$((MIN_FREE_GIB * 1024 * 1024 * 1024))" \
    "${RUN_NAME}" "${REMOTE_PYTEST_ROOT}" <<'REMOTE_PREFLIGHT'
set -euo pipefail
snapshot="$1"; result_root="$2"; data_root="$3"; support_rel="$4"
support_hash="$5"; min_free_bytes="$6"; run_name="$7"; pytest_root="$8"
test ! -e "${result_root}"
result_parent="$(dirname "${result_root}")"
mkdir -p "${result_parent}"
test -w "${result_parent}"
test -d "${pytest_root}"
test "$(sha256sum "${snapshot}/${support_rel}" | awk '{print $1}')" = "${support_hash}"
test "$(python3 -c 'import torch; print(torch.cuda.device_count())')" -eq 4
python3 -c 'import cv2, faiss, numpy, scipy, sklearn, tifffile, torch, yaml'
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
  src/post_eval.py fmad/evaluation/metrics.py fmad/methods/superad.py \
  scripts/aggregate_same16_baselines.py \
  scripts/aggregate_a6_fusion_public_shadow.py \
  scripts/adjudicate_same16_a6.py
PYTHONPATH="${pytest_root}" python3 -m pytest -q \
  tests/test_post_eval_numeric_stability.py \
  tests/test_flow_tte_postprocess_eval.py \
  tests/test_superad_fixed_reference.py \
  tests/test_aggregate_same16_baselines.py \
  tests/test_aggregate_a6_fusion_public_shadow.py \
  tests/test_adjudicate_same16_a6.py
python3 - "${support_rel}" "${data_root}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
data_root = Path(sys.argv[2])
objects = {"can", "fabric", "fruit_jelly", "rice", "sheet_metal", "vial", "wallplugs", "walnuts"}
if set(manifest) != objects:
    raise SystemExit("support object set mismatch")
for object_name in objects:
    paths = manifest[object_name]
    if len(paths) != 16 or len(set(paths)) != 16:
        raise SystemExit(f"{object_name}: invalid fixed-reference count")
    expected = data_root / object_name / "train" / "good"
    for value in paths:
        path = Path(value)
        if path.parent != expected or not path.is_file():
            raise SystemExit(f"invalid fixed reference: {path}")
PY
printf 'PREFLIGHT_OK run=%s result_parent=%s free_bytes=%s\n' \
  "${run_name}" "${result_parent}" "${free_bytes}"
REMOTE_PREFLIGHT
}

start_controller() {
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  ssh_remote "${REMOTE_HOST}" docker exec -d \
    -e RUN_NAME="${RUN_NAME}" \
    -e REMOTE_SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT}" \
    -e REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT}" \
    -e DATA_ROOT="${DATA_ROOT}" \
    -e TORCH_HOME=/workspace/.cache/torch \
    -e FMAD_DINOV2_OFFLINE=1 \
    "${FIXED_CONTAINER}" \
    bash "${snapshot}/${CONTROLLER_REL}" internal-controller
}

run_basic_object() {
  local snapshot="$1" result_root="$2" gpu="$3" object_name="$4"
  local output="${result_root}/basic/objects/${object_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${snapshot}" \
    python3 "${snapshot}/scripts/run_flow_tte_mvtec_ad2.py" \
      --data-root "${DATA_ROOT}" --project-root "${snapshot}" --fsad-root "${snapshot}" \
      --output-root "${output}" --objects "${object_name}" \
      --shots 16 --seed 0 --device cuda --flow-epochs 3 \
      --coupling-layers 2 --hidden-multiplier 1 --flow-lr 2e-4 --flow-clamp 1.9 \
      --flow-transform-mode flow --tail-weight 0.3 --tail-top-k-ratio 0.05 \
      --lambda-logdet 1e-3 --density-quantile 0.90 --expansion-budget 1.0 \
      --distance-weight 1.0 --density-weight 0.25 --score-mode latent_distance \
      --top-percent 0.01 --query-chunk-size 512 --calibration-sample-size 0 \
      --loo-standardization off --latent-bank-subsample none \
      --pro-integration-limit 0.05 \
      --rgb-guide none --threshold-calibration-mode none --binary-postprocess none \
      --backbone-model dinov2_vitl14 --feature-layers 5,11,17,23 \
      --support-selection "fixed_json=${snapshot}/${SUPPORT_REL}" \
      --support-selection-seed 0 --support-transforms identity \
      --feature-fusion layer_norm_mean --normality-mode fused \
      --dvt-denoise-mode position_mean --dvt-denoise-alpha 1.0 \
      --context-source none --flow-context-source auto --memory-context-source auto \
      --context-mode none --context-weight 0.0 --context-top-m 1 \
      --flow-condition-mode none --transformer-context-mode none
}

run_knn_object() {
  local snapshot="$1" result_root="$2" gpu="$3" object_name="$4" method="$5"
  local output="${result_root}/${method}/objects/${object_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${snapshot}" \
    python3 "${snapshot}/run.py" \
      --config "${snapshot}/configs/baselines/same16_${method}_ad2.yaml" \
      --data-root "${DATA_ROOT}" --output-dir "${output}" \
      --objects "${object_name}" --device cuda:0
  local run_dir="${output}/${method}/mvtec_ad2/dinov2_vitl14/seed=0"
  test -f "${run_dir}/metrics.json"
  test -f "${run_dir}/run_manifest.json"
}

run_mechanism_object() {
  local snapshot="$1" result_root="$2" gpu="$3" object_name="$4" arm="$5"
  local layers normality transform_mode
  case "${arm}" in
    a6)
      layers="5,11,17,23"; normality="fused"; transform_mode="flow"
      ;;
    a7)
      layers="5,11,17,23"; normality="raw_nn"; transform_mode="identity"
      ;;
    a6_l17)
      layers="17"; normality="fused"; transform_mode="flow"
      ;;
    l17)
      layers="17"; normality="raw_nn"; transform_mode="identity"
      ;;
    *)
      die "unknown mechanism arm: ${arm}"
      ;;
  esac
  local output="${result_root}/mechanism/arms/${arm}/objects/${object_name}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${snapshot}" \
    python3 "${snapshot}/scripts/run_flow_tte_mvtec_ad2.py" \
      --data-root "${DATA_ROOT}" --project-root "${snapshot}" --fsad-root "${snapshot}" \
      --output-root "${output}" --objects "${object_name}" \
      --shots 16 --seed 0 --device cuda --flow-epochs 3 \
      --coupling-layers 2 --hidden-multiplier 1 --flow-lr 2e-4 --flow-clamp 1.9 \
      --flow-transform-mode "${transform_mode}" --tail-weight 0.3 \
      --tail-top-k-ratio 0.05 --lambda-logdet 1e-3 \
      --density-quantile 0.90 --expansion-budget 1.0 \
      --distance-weight 1.0 --density-weight 0.0 --score-mode latent_distance \
      --top-percent 0.01 --query-chunk-size 512 --calibration-sample-size 0 \
      --loo-standardization off --latent-bank-subsample none \
      --pro-integration-limit 0.05 \
      --rgb-guide none --threshold-calibration-mode none --binary-postprocess none \
      --backbone-model dinov2_vitl14 --feature-layers "${layers}" \
      --support-selection "fixed_json=${snapshot}/${SUPPORT_REL}" \
      --support-selection-seed 0 --support-transforms identity \
      --feature-fusion layer_norm_mean --normality-mode "${normality}" \
      --dvt-denoise-mode none --dvt-denoise-alpha 1.0 \
      --context-source none --flow-context-source auto --memory-context-source auto \
      --context-mode none --context-weight 0.0 --context-top-m 1 \
      --flow-condition-mode none --transformer-context-mode none
}

run_group() {
  local snapshot="$1" result_root="$2" gpu="$3"
  shift 3
  local object_name method
  for object_name in "$@"; do
    mkdir -p "${result_root}/logs/gpu${gpu}"
    run_basic_object "${snapshot}" "${result_root}" "${gpu}" "${object_name}" \
      >"${result_root}/logs/gpu${gpu}/${object_name}_basic.log" 2>&1
    for method in superad rn_fmlk; do
      run_knn_object "${snapshot}" "${result_root}" "${gpu}" "${object_name}" "${method}" \
        >"${result_root}/logs/gpu${gpu}/${object_name}_${method}.log" 2>&1
    done
    for method in a6 a7 a6_l17 l17; do
      run_mechanism_object "${snapshot}" "${result_root}" "${gpu}" "${object_name}" "${method}" \
        >"${result_root}/logs/gpu${gpu}/${object_name}_${method}.log" 2>&1
    done
  done
}

internal_controller() {
  require_run_name
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  test -d "${snapshot}"
  test ! -e "${result_root}"
  mkdir -p "${result_root}/logs"
  exec > >(tee -a "${result_root}/controller.log") 2>&1
  printf '%s\n' "$$" >"${result_root}/controller.pid"
  cp "${snapshot}/snapshot_hash.txt" "${result_root}/snapshot_hash.txt"
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
    printf 'FAILED\n' >"${result_root}/failed.txt"
    exit 1
  fi
  if ! python3 "${snapshot}/${AGGREGATOR_REL}" \
    --root "${result_root}" --support-manifest "${snapshot}/${SUPPORT_REL}" \
    >"${result_root}/aggregate.log" 2>&1; then
    printf 'FAILED aggregate_same16_baselines\n' >"${result_root}/failed.txt"
    exit 1
  fi
  if ! PYTHONPATH="${snapshot}" python3 "${snapshot}/${A6_AGGREGATOR_REL}" \
    --root "${result_root}" --support-manifest "${snapshot}/${SUPPORT_REL}" \
    >"${result_root}/a6_fusion_aggregate.log" 2>&1; then
    printf 'FAILED aggregate_a6_fusion_public_shadow\n' >"${result_root}/failed.txt"
    exit 1
  fi
  if ! PYTHONPATH="${snapshot}" python3 "${snapshot}/${ADJUDICATOR_REL}" \
    --root "${result_root}" >"${result_root}/post_a6_decision.log" 2>&1; then
    printf 'FAILED adjudicate_same16_a6\n' >"${result_root}/failed.txt"
    exit 1
  fi
  local decision_marker_count
  decision_marker_count="$(
    find "${result_root}" -maxdepth 1 -type f \
      \( -name shot_scaling_authorized.txt -o -name shot_scaling_blocked.txt \) \
      | wc -l
  )"
  test "${decision_marker_count}" -eq 1
  while IFS= read -r maps_dir; do
    find "${maps_dir}" -type f -delete
    find "${maps_dir}" -depth -type d -empty -delete
  done < <(
    find "${result_root}" -depth -type d \
      \( -name anomaly_maps -o -name calibration_maps \) -print
  )
  if find "${result_root}" -type d \
    \( -name anomaly_maps -o -name calibration_maps \) -print -quit | grep -q .; then
    printf 'dense maps remain\n' >"${result_root}/failed.txt"
    exit 1
  fi
  printf 'dense_anomaly_and_calibration_map_directories=0\n' \
    >"${result_root}/cleanup_evidence.txt"
  printf 'completed_utc=%s\n' "$(date -u +%FT%TZ)" >"${result_root}/complete.txt"
  cd /workspace
  find "${snapshot}" -type f -delete
  find "${snapshot}" -depth -type d -empty -delete
  printf 'source_snapshot_removed=true\n' >>"${result_root}/cleanup_evidence.txt"
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
  printf 'STARTED run=%s\n' "${RUN_NAME}"
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
  test -f "${root}/summary.tsv" && cat "${root}/summary.tsv"
  test -f "${root}/post_a6_decision.json" && cat "${root}/post_a6_decision.json"
  test -f "${root}/shot_scaling_authorized.txt" \
    && cat "${root}/shot_scaling_authorized.txt"
  test -f "${root}/shot_scaling_blocked.txt" \
    && cat "${root}/shot_scaling_blocked.txt"
elif [[ -f "${root}/failed.txt" ]]; then
  cat "${root}/failed.txt"
  exit 1
elif [[ -f "${root}/controller.pid" ]]; then
  pid="$(cat "${root}/controller.pid")"
  printf 'RUNNING pid=%s alive=%s\n' "${pid}" "$(kill -0 "${pid}" 2>/dev/null && echo true || echo false)"
  find "${root}" -name metrics.json -type f | wc -l
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
  test -f "${destination}/summary.json"
  test -f "${destination}/a6_fusion_summary.json"
  test -f "${destination}/post_a6_decision.json"
  test -f "${destination}/cleanup_evidence.txt"
  local decision_marker_count
  decision_marker_count="$(
    find "${destination}" -maxdepth 1 -type f \
      \( -name shot_scaling_authorized.txt -o -name shot_scaling_blocked.txt \) \
      | wc -l
  )"
  test "${decision_marker_count}" -eq 1
  printf 'PULLED_EVIDENCE_ONLY destination=%s\n' "${destination}"
  printf 'Run authorize-shot-scaling before any shot continuation.\n'
}

authorize_shot_scaling() {
  require_run_name
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}"
  test -f "${destination}/complete.txt" \
    || die "missing completed local pull: ${destination}"
  test -f "${destination}/post_a6_decision.json" \
    || die "missing post-A6 decision: ${destination}"
  if [[ -f "${destination}/shot_scaling_blocked.txt" ]]; then
    die "shot scaling is blocked by the preregistered decision"
  fi
  test -f "${destination}/shot_scaling_authorized.txt" \
    || die "shot scaling has no authorization marker"
  printf 'SHOT_SCALING_AUTHORIZED run=%s\n' "${RUN_NAME}"
  cat "${destination}/shot_scaling_authorized.txt"
}

case "${MODE}" in
  start) start ;;
  status) status ;;
  pull) pull ;;
  authorize-shot-scaling) authorize_shot_scaling ;;
  internal-controller) internal_controller ;;
  *) die "usage: $0 {start|status|pull|authorize-shot-scaling|internal-controller}" ;;
esac
