#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-status}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRESET="${ROOT_DIR}/configs/remote/dsba5.env"
ASKPASS_SCRIPT="${ROOT_DIR}/scripts/ssh_askpass_env.sh"
FIXED_CONTAINER="hun_fsad_tta_012"
REMOTE_SNAPSHOT_ROOT="${REMOTE_SNAPSHOT_ROOT:-/workspace/isolated_runs}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/workspace/results_remote}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
SUPPORT_REL="skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json"
LOCAL_PULL_ROOT="${LOCAL_PULL_ROOT:-${ROOT_DIR}/results/remote_runs/dsba5}"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

require_run_name() {
  [[ -n "${RUN_NAME:-}" ]] || die "RUN_NAME is required for ${MODE}"
  [[ "${RUN_NAME}" =~ ^wtcf_stage_minus1_[A-Za-z0-9._-]+$ ]] || die "unsafe WTCF RUN_NAME"
}

load_preset() {
  [[ -f "${PRESET}" ]] || die "missing preset: ${PRESET}"
  # shellcheck disable=SC1090
  source "${PRESET}"
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
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    test ! -e \"${root}\"
    test \"\$(python3 -c \"import torch; print(torch.cuda.device_count())\")\" -eq 1
    python3 -c \"import cv2,numpy,sklearn,torch\"
    test -f '${snapshot}/${SUPPORT_REL}'
    for object_name in can fruit_jelly rice fabric sheet_metal; do
      test -d '${DATA_ROOT}/'\"\${object_name}\"'/train/good'
    done
    python3 -m py_compile \
      '${snapshot}/src/flow_tte/wtcf.py' \
      '${snapshot}/scripts/run_wtcf_stage_minus1.py' \
      '${snapshot}/scripts/aggregate_wtcf_stage_minus1.py'
    PYTHONPATH='${snapshot}:${snapshot}/src:${snapshot}/scripts' \
      python3 '${snapshot}/scripts/smoke_wtcf_core.py' \
      >'/tmp/${RUN_NAME}_core_smoke.log'
    PYTHONPATH='${snapshot}:${snapshot}/src:${snapshot}/scripts' \
      python3 '${snapshot}/scripts/run_wtcf_stage_minus1.py' --help \
      >'/tmp/${RUN_NAME}_runner_help.log'
    TORCH_HOME=/workspace/.cache/torch FMAD_DINOV2_OFFLINE=1 \
      PYTHONPATH='${snapshot}:${snapshot}/src' \
      python3 '${snapshot}/scripts/preflight_dinov2_cache.py' --model dinov2_vitl14 \
      >'/tmp/${RUN_NAME}_dinov2.log'
    mkdir -p '${root}/cases'
    cp '/tmp/${RUN_NAME}_core_smoke.log' '${root}/preflight_core_smoke.log'
    cp '/tmp/${RUN_NAME}_runner_help.log' '${root}/preflight_runner_help.log'
    cp '/tmp/${RUN_NAME}_dinov2.log' '${root}/preflight_dinov2.log'
    nvidia-smi --query-gpu=index,uuid,memory.total --format=csv,noheader >'${root}/gpu_visibility.txt'
    printf 'status=passed\\ncontainer_cuda_count=1\\nhost_gpu_id=3\\ncontainer_cuda_slot=0\\n' \
      >'${root}/preflight.txt'
  '"
}

run_case() {
  local object_name="$1" seed="$2"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local case_root="${root}/cases/${object_name}_seed${seed}"
  CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 \
    TORCH_HOME=/workspace/.cache/torch FMAD_DINOV2_OFFLINE=1 \
    PYTHONPATH="${snapshot}:${snapshot}/src:${snapshot}/scripts" \
    python3 "${snapshot}/scripts/run_wtcf_stage_minus1.py" \
      --data-root "${DATA_ROOT}" --fixed-support-json "${snapshot}/${SUPPORT_REL}" \
      --output-root "${case_root}" --object "${object_name}" --support-seed "${seed}" \
      --device cuda:0 --host-gpu-id 3 --container-cuda-slot 0 \
      >"${root}/cases/${object_name}_seed${seed}.log" 2>&1
}

check_sentinel() {
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  python3 - "${root}/cases/can_seed0/summary.json" "${root}/sentinel_gate.json" <<'PY'
import json, sys
source, target = sys.argv[1:]
case = json.load(open(source, encoding="utf-8"))
arm = case["arms"]["wtcf"]
training = arm["training"]
checks = {
    "finite_nonconstant_complete": arm["finite_nonconstant_complete"] is True,
    "soft_hard_proxy": arm["proxy"]["passed"] is True,
    "training_finite": training["finite"] is True,
    "function_geometry": training["final_geometry"]["passed"] is True,
    "jvp_directional": training["final_directional"]["passed"] is True,
    "nonzero_stable_witness": training["witness"]["accepted_count"] > 0,
    "access_custody": case["access_ledger"]["coverage"] == 1.0 and case["access_ledger"]["public_private_test_ground_truth_attempts"] == 0,
}
payload = {"checks": checks, "passed": all(checks.values())}
with open(target, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
raise SystemExit(0 if payload["passed"] else 17)
PY
}

internal_controller() {
  require_run_name
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  trap 'code=$?; if ((code!=0)) && [[ ! -f "'"${root}"'/failed.txt" ]]; then printf "exit_code=%s\n" "${code}" >"'"${root}"'/failed.txt"; fi' EXIT
  exec 9>"${REMOTE_SNAPSHOT_ROOT}/.wtcf_stage_minus1_gpu3.lock"
  flock -n 9 || { echo cooperative_lock_busy >"${root}/failed.txt"; return 75; }
  printf 'started_utc=%s\n' "$(date -u +%FT%TZ)" >"${root}/started.txt"
  run_case can 0
  if ! check_sentinel; then
    printf 'EARLY_VALIDITY_KILL_NO_FULL_STAGE\n' >"${root}/verdict.txt"
    printf 'status=complete_early_kill\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" >"${root}/complete.txt"
    return 0
  fi
  for object_name in can fruit_jelly rice fabric sheet_metal; do
    for seed in 0 1 2; do
      [[ "${object_name}" == can && "${seed}" == 0 ]] && continue
      run_case "${object_name}" "${seed}"
    done
  done
  PYTHONPATH="${snapshot}:${snapshot}/src" python3 \
    "${snapshot}/scripts/aggregate_wtcf_stage_minus1.py" --root "${root}" \
    >"${root}/aggregate.log" 2>&1
  cp "${root}/verdict.txt" "${root}/protocol_verdict.txt"
  printf 'status=complete\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" >"${root}/complete.txt"
}

start_remote() {
  require_run_name
  local archive hash
  archive="$(mktemp "${TMPDIR:-/tmp}/wtcf-source.XXXXXX.tar")"
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
      bash \"\${snapshot}/scripts/run_wtcf_stage_minus1_remote.sh\" internal-controller \
      >\"\${root}/controller.log\" 2>&1 </dev/null &
    printf "%s\\n" \"\$!\" >\"\${root}/controller.pid\"
    printf "STARTED run_name=%s pid=%s\\n" '${RUN_NAME}' \"\$!\"
  '"
}

status_remote() {
  require_run_name
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    test -f \"\${root}/complete.txt\" && status=complete || \
      { test -f \"\${root}/failed.txt\" && status=failed || status=running; }
    echo run_name=${RUN_NAME}; echo status=\"\${status}\"
    echo case_summaries=\"\$(find \"\${root}/cases\" -name summary.json 2>/dev/null | wc -l)\"
    test -f \"\${root}/sentinel_gate.json\" && cat \"\${root}/sentinel_gate.json\" || true
    test -f \"\${root}/failed.txt\" && cat \"\${root}/failed.txt\" || true
    test -f \"\${root}/verdict.txt\" && cat \"\${root}/verdict.txt\" || true
    echo controller_log_tail
    test -f \"\${root}/controller.log\" && tail -n 12 \"\${root}/controller.log\" || true
    echo active_case_log_tail
    case_log=\$(find \"\${root}/cases\" -maxdepth 1 -name '*.log' 2>/dev/null | sort | tail -n 1)
    test -n \"\${case_log}\" && tail -n 12 \"\${case_log}\" || true
  '"
  ssh_remote "${REMOTE_HOST}" \
    "nvidia-smi -i 3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
}

pull_remote() {
  require_run_name
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}"
  [[ ! -e "${destination}" ]] || die "destination exists: ${destination}"
  mkdir -p "${destination}"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' tar -C \
    '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' -cf - ." | tar -C "${destination}" -xf -
  test -f "${destination}/complete.txt"
  printf 'PULLED %s\n' "${destination}"
}

cleanup_remote() {
  require_run_name
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}"
  [[ "${snapshot}" == /workspace/isolated_runs/wtcf_stage_minus1_* ]] || die "unsafe cleanup"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    test -f '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/complete.txt'
    rm -rf -- '${snapshot}'
  '"
}

invalidate_remote() {
  require_run_name
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    test -f \"\${root}/controller.pid\"
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
    printf 'reason=implementation_changed_before_scientific_result\\ninvalidated_utc=%s\\n' \
      \"\$(date -u +%FT%TZ)\" >\"\${root}/invalidated.txt\"
  '"
}

case "${MODE}" in
  internal-controller) internal_controller ;;
  start) load_preset; start_remote ;;
  status) load_preset; status_remote ;;
  pull) load_preset; pull_remote ;;
  cleanup) load_preset; cleanup_remote ;;
  invalidate) load_preset; invalidate_remote ;;
  *) die "unknown mode: ${MODE}" ;;
esac
