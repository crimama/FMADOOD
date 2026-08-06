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
STAGE1_EVALUATOR_PREOPT_SHA256="03e69096dd4014d7e76d3df67e8cfdd389e16de3028297b0c2abbbe995983857"
STAGE1_EVALUATOR_OPT1_SHA256="daf0413225de19d8279ad1cf66e41dc741036cbb7b33857a2de3a10e0d290b52"
STAGE1_EVALUATOR_OPT2_SHA256="0325ccc9163548439ccf5f6632a545a8505e0c60f04897776946562088978775"
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
  [[ "${RUN_NAME}" =~ ^default_shift_sequence_[A-Za-z0-9._-]+$ ]] \
    || die "RUN_NAME must start with default_shift_sequence_"
}

load_remote_preset() {
  [[ -f "${PRESET}" ]] || die "missing local preset: ${PRESET}"
  # shellcheck disable=SC1090
  source "${PRESET}"
  [[ -n "${REMOTE_HOST:-}" && -n "${REMOTE_PORT:-}" ]] \
    || die "remote preset is missing REMOTE_HOST or REMOTE_PORT"
  [[ "${CONTAINER_NAME:-${FIXED_CONTAINER}}" == "${FIXED_CONTAINER}" ]] \
    || die "this sequence must use ${FIXED_CONTAINER}"
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
  if (( memory > memory_limit || utilization > util_limit )); then
    printf 'GPU_BUSY gpu=%s memory_mib=%s utilization=%s\n' \
      "${gpu}" "${memory}" "${utilization}" >&2
    exit 73
  fi
  processes="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid \
    --format=csv,noheader,nounits)"
  if grep -Eq '[0-9]+' <<<"${processes}"; then
    printf 'GPU_COMPUTE_PROCESS_PRESENT gpu=%s\n' "${gpu}" >&2
    exit 74
  fi
  printf 'GPU_IDLE gpu=%s memory_mib=%s utilization=%s compute_apps=0\n' \
    "${gpu}" "${memory}" "${utilization}"
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

stream_stage2_snapshot() {
  local archive="$1" archive_hash="$2"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo"
  ssh_remote "${REMOTE_HOST}" \
    "docker exec -i '${FIXED_CONTAINER}' bash -lc '
      set -euo pipefail
      snapshot=\"${snapshot}\"
      archive=/tmp/${RUN_NAME}_stage2_source.tar
      test ! -e \"\${snapshot}\"
      mkdir -p \"\${snapshot}\"
      tee \"\${archive}\" >/dev/null
      actual=\$(sha256sum \"\${archive}\")
      actual=\"\${actual%% *}\"
      test \"\${actual}\" = \"${archive_hash}\"
      tar -C \"\${snapshot}\" -xf \"\${archive}\"
      rm -f \"\${archive}\"
      printf \"archive_sha256=%s\\n\" \"\${actual}\" >\"\${snapshot}/snapshot_hash.txt\"
    '" <"${archive}"
}

container_preflight() {
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local runner_hash evaluator_hash guide_hash controller_hash
  runner_hash="$(sha256sum "${ROOT_DIR}/scripts/run_flow_tte_mvtec_ad2.py" | awk '{print $1}')"
  evaluator_hash="$(sha256sum "${ROOT_DIR}/scripts/evaluate_flow_tte_default_shift_audit.py" | awk '{print $1}')"
  guide_hash="$(sha256sum "${ROOT_DIR}/scripts/run_flow_tte_mvtecad2_guided_refinement.py" | awk '{print $1}')"
  controller_hash="$(sha256sum "$0" | awk '{print $1}')"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    snapshot=${snapshot}
    result_root=${result_root}
    test ! -e \"\${result_root}\"
    command -v flock >/dev/null
    python3 -c \"import cv2, numpy, sklearn, tifffile, torch\"
    test \"\$(python3 -c \"import torch; print(torch.cuda.device_count())\")\" -eq 4
    free_bytes=\$(df -PB1 /workspace | awk \"NR==2 {print \\\$4}\")
    test \"\${free_bytes}\" -ge $((MIN_FREE_GIB * 1024 * 1024 * 1024))
    test \"\$(sha256sum \"\${snapshot}/${SUPPORT_REL}\" | awk \"{print \\\$1}\")\" = \"${SUPPORT_SHA256}\"
    test \"\$(sha256sum \"\${snapshot}/scripts/run_flow_tte_mvtec_ad2.py\" | awk \"{print \\\$1}\")\" = \"${runner_hash}\"
    test \"\$(sha256sum \"\${snapshot}/scripts/evaluate_flow_tte_default_shift_audit.py\" | awk \"{print \\\$1}\")\" = \"${evaluator_hash}\"
    test \"\$(sha256sum \"\${snapshot}/scripts/run_flow_tte_mvtecad2_guided_refinement.py\" | awk \"{print \\\$1}\")\" = \"${guide_hash}\"
    test \"\$(sha256sum \"\${snapshot}/scripts/run_default_shift_sequence_remote.sh\" | awk \"{print \\\$1}\")\" = \"${controller_hash}\"
    test -d ${DATA_ROOT}
    for object_name in can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts; do
      test -d ${DATA_ROOT}/\${object_name}/train/good
      test -d ${DATA_ROOT}/\${object_name}/test_public/good
      test -d ${DATA_ROOT}/\${object_name}/test_public/bad
    done
    PYTHONPATH=\"\${snapshot}/scripts\" TRANSFORMERS_OFFLINE=1 python3 -c \"import sys; from dinov3_backbone import patch_torch_pytree_compat; patch_torch_pytree_compat(); from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)\" facebook/dinov3-vith16plus-pretrain-lvd1689m
    mkdir -p \"\${result_root}\"
    {
      printf \"run_name=%s\\n\" ${RUN_NAME}
      cat \"\${snapshot}/snapshot_hash.txt\"
      printf \"runner_sha256=%s\\nevaluator_sha256=%s\\nguide_sha256=%s\\ncontroller_sha256=%s\\n\" \\
        ${runner_hash} ${evaluator_hash} ${guide_hash} ${controller_hash}
      printf \"support_manifest_sha256=%s\\nfree_bytes=%s\\ncontainer_cuda_count=4\\n\" \\
        ${SUPPORT_SHA256} \"\${free_bytes}\"
    } >\"\${result_root}/preflight.txt\"
  '"
}

start_stage1() {
  require_run_name
  assert_fixed_container
  local archive archive_hash
  archive="$(mktemp "${TMPDIR:-/tmp}/default-shift-source.XXXXXX.tar")"
  trap 'rm -f -- "'"${archive}"'"' EXIT
  make_snapshot_archive "${archive}"
  archive_hash="$(sha256sum "${archive}" | awk '{print $1}')"
  stream_isolated_snapshot "${archive}" "${archive_hash}"
  container_preflight
  host_gpu_idle_check
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    snapshot=${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo
    result_root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    nohup env RUN_NAME=${RUN_NAME} REMOTE_SNAPSHOT_ROOT=${REMOTE_SNAPSHOT_ROOT} \\
      REMOTE_RESULTS_ROOT=${REMOTE_RESULTS_ROOT} DATA_ROOT=${DATA_ROOT} \\
      TORCH_HOME=/workspace/.cache/torch FMAD_DINOV3_OFFLINE=1 \\
      bash \"\${snapshot}/scripts/run_default_shift_sequence_remote.sh\" internal-stage1 \\
      >\"\${result_root}/controller_stage1.log\" 2>&1 </dev/null &
    printf \"%s\\n\" \"\$!\" >\"\${result_root}/controller_stage1.pid\"
    printf \"STARTED run_name=%s pid=%s archive_sha256=%s\\n\" \\
      ${RUN_NAME} \"\$!\" ${archive_hash}
  '"
  rm -f -- "${archive}"
  trap - EXIT
}

container_preflight_stage2() {
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo"
  local sequence_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local result_root="${sequence_root}/stage2"
  local runner_hash evaluator_hash
  runner_hash="$(sha256sum "${ROOT_DIR}/scripts/run_flow_tte_support_bank_routing.py" | awk '{print $1}')"
  evaluator_hash="$(sha256sum "${ROOT_DIR}/scripts/evaluate_flow_tte_support_bank_routing.py" | awk '{print $1}')"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    snapshot=${snapshot}
    sequence_root=${sequence_root}
    result_root=${result_root}
    test -f \"\${sequence_root}/stage1_complete.txt\"
    test ! -e \"\${result_root}\"
    python3 -c \"import cv2, numpy, sklearn, tifffile, torch\"
    test \"\$(python3 -c \"import torch; print(torch.cuda.device_count())\")\" -eq 4
    support_actual=\$(sha256sum \"\${snapshot}/${SUPPORT_REL}\")
    runner_actual=\$(sha256sum \"\${snapshot}/scripts/run_flow_tte_support_bank_routing.py\")
    evaluator_actual=\$(sha256sum \"\${snapshot}/scripts/evaluate_flow_tte_support_bank_routing.py\")
    test \"\${support_actual%% *}\" = \"${SUPPORT_SHA256}\"
    test \"\${runner_actual%% *}\" = \"${runner_hash}\"
    test \"\${evaluator_actual%% *}\" = \"${evaluator_hash}\"
    PYTHONPATH=\"\${snapshot}/scripts\" TRANSFORMERS_OFFLINE=1 python3 -c \"import sys; from dinov3_backbone import patch_torch_pytree_compat; patch_torch_pytree_compat(); from transformers import AutoConfig; AutoConfig.from_pretrained(sys.argv[1], local_files_only=True)\" facebook/dinov3-vith16plus-pretrain-lvd1689m
    mkdir -p \"\${result_root}/logs\"
    {
      printf \"run_name=%s\\nstage=2\\n\" ${RUN_NAME}
      cat \"\${snapshot}/snapshot_hash.txt\"
      printf \"runner_sha256=%s\\nevaluator_sha256=%s\\nsupport_manifest_sha256=%s\\n\" \\
        ${runner_hash} ${evaluator_hash} ${SUPPORT_SHA256}
    } >\"\${result_root}/preflight.txt\"
  '"
}

start_stage2() {
  require_run_name
  assert_fixed_container
  local archive archive_hash
  archive="$(mktemp "${TMPDIR:-/tmp}/default-shift-stage2.XXXXXX.tar")"
  trap 'rm -f -- "'"${archive}"'"' EXIT
  make_snapshot_archive "${archive}"
  archive_hash="$(sha256sum "${archive}" | awk '{print $1}')"
  stream_stage2_snapshot "${archive}" "${archive_hash}"
  container_preflight_stage2
  host_gpu_idle_check
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    snapshot=${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo
    result_root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage2
    nohup env RUN_NAME=${RUN_NAME} REMOTE_SNAPSHOT_ROOT=${REMOTE_SNAPSHOT_ROOT} \\
      REMOTE_RESULTS_ROOT=${REMOTE_RESULTS_ROOT} DATA_ROOT=${DATA_ROOT} \\
      TORCH_HOME=/workspace/.cache/torch FMAD_DINOV3_OFFLINE=1 \\
      bash \"\${snapshot}/scripts/run_default_shift_sequence_remote.sh\" internal-stage2 \\
      >\"\${result_root}/controller.log\" 2>&1 </dev/null &
    printf \"%s\\n\" \"\$!\" >\"\${result_root}/controller.pid\"
    printf \"STARTED stage=2 run_name=%s pid=%s archive_sha256=%s\\n\" \\
      ${RUN_NAME} \"\$!\" ${archive_hash}
  '"
  rm -f -- "${archive}"
  trap - EXIT
}

start_stage3() {
  require_run_name
  assert_fixed_container
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    snapshot=${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo
    sequence_root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    result_root=\"\${sequence_root}/stage3\"
    test -f \"\${sequence_root}/stage1_complete.txt\"
    test -f \"\${sequence_root}/stage2/complete.txt\"
    test -x \"\${snapshot}/scripts/evaluate_flow_tte_guided_fallback.py\"
    test ! -e \"\${result_root}\"
    mkdir -p \"\${result_root}\"
    nohup env RUN_NAME=${RUN_NAME} REMOTE_SNAPSHOT_ROOT=${REMOTE_SNAPSHOT_ROOT} \\
      REMOTE_RESULTS_ROOT=${REMOTE_RESULTS_ROOT} DATA_ROOT=${DATA_ROOT} \\
      bash \"\${snapshot}/scripts/run_default_shift_sequence_remote.sh\" internal-stage3 \\
      >\"\${result_root}/controller.log\" 2>&1 </dev/null &
    printf \"%s\\n\" \"\$!\" >\"\${result_root}/controller.pid\"
    printf \"STARTED stage=3 run_name=%s pid=%s\\n\" ${RUN_NAME} \"\$!\"
  '"
}

start_stage4() {
  require_run_name
  assert_fixed_container
  local runner_hash evaluator_hash
  runner_hash="$(sha256sum "${ROOT_DIR}/scripts/run_flow_tte_local_soft_nuisance.py" | awk '{print $1}')"
  evaluator_hash="$(sha256sum "${ROOT_DIR}/scripts/evaluate_flow_tte_local_soft_nuisance.py" | awk '{print $1}')"
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    snapshot=${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo
    sequence_root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    result_root=\"\${sequence_root}/stage4\"
    test -f \"\${sequence_root}/stage3/complete.txt\"
    test ! -e \"\${result_root}\"
    runner_actual=\$(sha256sum \"\${snapshot}/scripts/run_flow_tte_local_soft_nuisance.py\")
    evaluator_actual=\$(sha256sum \"\${snapshot}/scripts/evaluate_flow_tte_local_soft_nuisance.py\")
    test \"\${runner_actual%% *}\" = ${runner_hash}
    test \"\${evaluator_actual%% *}\" = ${evaluator_hash}
    mkdir -p \"\${result_root}/logs\"
    printf \"runner_sha256=%s\\nevaluator_sha256=%s\\n\" \\
      ${runner_hash} ${evaluator_hash} >\"\${result_root}/preflight.txt\"
  '"
  host_gpu_idle_check
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    set -euo pipefail
    snapshot=${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo
    result_root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage4
    nohup env RUN_NAME=${RUN_NAME} REMOTE_SNAPSHOT_ROOT=${REMOTE_SNAPSHOT_ROOT} \\
      REMOTE_RESULTS_ROOT=${REMOTE_RESULTS_ROOT} DATA_ROOT=${DATA_ROOT} \\
      TORCH_HOME=/workspace/.cache/torch FMAD_DINOV3_OFFLINE=1 \\
      bash \"\${snapshot}/scripts/run_default_shift_sequence_remote.sh\" internal-stage4 \\
      >\"\${result_root}/controller.log\" 2>&1 </dev/null &
    printf \"%s\\n\" \"\$!\" >\"\${result_root}/controller.pid\"
    printf \"STARTED stage=4 run_name=%s pid=%s\\n\" ${RUN_NAME} \"\$!\"
  '"
}

run_default_arm() {
  local gpu="$1" object_name="$2" arm="$3" density_weight="$4"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local output_root="${result_root}/objects/${object_name}/${arm}"
  mkdir -p "${result_root}/logs"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${snapshot}:${snapshot}/src" \
    python3 "${snapshot}/scripts/run_flow_tte_mvtec_ad2.py" \
      --data-root "${DATA_ROOT}" \
      --output-root "${output_root}" \
      --project-root /workspace \
      --fsad-root "${snapshot}" \
      --objects "${object_name}" \
      --shots 16 --seed 0 --device cuda \
      --flow-epochs 3 --coupling-layers 2 --hidden-multiplier 1 \
      --flow-lr 2e-4 --flow-clamp 1.9 --flow-transform-mode flow \
      --tail-weight 0.3 --tail-top-k-ratio 0.05 --lambda-logdet 1e-3 \
      --density-quantile 0.90 --expansion-budget 1.0 \
      --distance-weight 1.0 --density-weight "${density_weight}" \
      --score-mode latent_distance --top-percent 0.01 --query-chunk-size 512 \
      --calibration-sample-size 0 --loo-standardization on \
      --latent-bank-subsample none --pro-integration-limit 0.05 \
      --rgb-guide none --binary-postprocess none \
      --morphology-line-length 17 --morphology-angle-count 16 \
      --backbone-model dinov3_vith16plus --backbone-resolution 0 \
      --feature-layers 7,15,23,31 --tile-patch-size 0 --tile-overlap 0 \
      --image-resize-factor 1.0 --support-brightness-range 1.0,1.0 \
      --support-selection "fixed_json=${snapshot}/${SUPPORT_REL}" \
      --support-selection-seed 0 --support-transforms identity \
      --feature-fusion layer_norm_mean --normality-mode fused \
      --residual-weight 0.25 --dvt-denoise-mode position_mean \
      --dvt-denoise-alpha 1.0 --context-source none \
      --flow-context-source auto --memory-context-source auto \
      --context-mode none --context-weight 0.0 --context-top-m 1 \
      --flow-condition-mode none --transformer-context-mode none \
      --score-field-calibration-mode none \
      --score-field-calibration-alpha 1.0 \
      --score-field-position-std-floor 0.25 \
      --score-field-foreground-mode none \
      --score-field-foreground-quantile 0.20 \
      --score-field-background-multiplier 0.50 \
      --score-field-foreground-smooth-kernel 5 \
      --score-field-support-score-quantile 0.90 \
      >"${result_root}/logs/gpu${gpu}_${object_name}_${arm}.log" 2>&1
}

run_guided_arm() {
  local gpu="$1" object_name="$2"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${snapshot}:${snapshot}/src" \
    python3 "${snapshot}/scripts/run_flow_tte_mvtecad2_guided_refinement.py" \
      --data-root "${DATA_ROOT}" \
      --source-root "${result_root}/objects/${object_name}/a1_density" \
      --output-root "${result_root}/objects/${object_name}/a2_guided" \
      --objects "${object_name}" --seed 0 --binary-postprocess none \
      --morphology-line-length 17 --morphology-angle-count 16 \
      >"${result_root}/logs/gpu${gpu}_${object_name}_a2_guided.log" 2>&1
}

run_stage1_object() {
  local gpu="$1" object_name="$2"
  run_default_arm "${gpu}" "${object_name}" a0_distance 0.0
  run_default_arm "${gpu}" "${object_name}" a1_density 0.25
  run_guided_arm "${gpu}" "${object_name}"
  printf 'gpu=%s object=%s stage1=complete\n' "${gpu}" "${object_name}" \
    >"${REMOTE_RESULTS_ROOT}/${RUN_NAME}/objects/${object_name}/stage1_complete.txt"
}

run_stage1_group() {
  local gpu="$1"
  shift
  local object_name
  for object_name in "$@"; do
    run_stage1_object "${gpu}" "${object_name}"
  done
  printf 'gpu=%s status=complete objects=%s\n' "${gpu}" "$*" \
    >"${REMOTE_RESULTS_ROOT}/${RUN_NAME}/gpu${gpu}_stage1_complete.txt"
}

internal_stage1() {
  require_run_name
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  record_failure() {
    local exit_code="$?"
    if [[ "${exit_code}" -ne 0 && ! -f "${result_root}/stage1_complete.txt" ]]; then
      printf 'stage1 controller failed with exit_code=%s\n' "${exit_code}" \
        >"${result_root}/stage1_failed.txt"
    fi
  }
  trap record_failure EXIT
  exec 9>"${REMOTE_SNAPSHOT_ROOT}/.default_shift_sequence_gpus_0_3.lock"
  flock -n 9 || { printf 'cooperative GPU lock is held\n' \
    >"${result_root}/stage1_failed.txt"; return 75; }
  printf 'controller_pid=%s\nstarted_utc=%s\n' "$$" "$(date -u +%FT%TZ)" \
    >"${result_root}/stage1_started.txt"

  read -r -a group0 <<<"${OBJECT_GROUP_0}"
  read -r -a group1 <<<"${OBJECT_GROUP_1}"
  read -r -a group2 <<<"${OBJECT_GROUP_2}"
  read -r -a group3 <<<"${OBJECT_GROUP_3}"
  run_stage1_group 0 "${group0[@]}" & p0=$!
  run_stage1_group 1 "${group1[@]}" & p1=$!
  run_stage1_group 2 "${group2[@]}" & p2=$!
  run_stage1_group 3 "${group3[@]}" & p3=$!
  status=0
  for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
    wait "${pid}" || status=1
  done
  if [[ "${status}" -ne 0 ]]; then
    printf 'one or more GPU groups failed\n' >"${result_root}/stage1_failed.txt"
    return 1
  fi
  PYTHONPATH="${snapshot}:${snapshot}/src" \
    python3 "${snapshot}/scripts/evaluate_flow_tte_default_shift_audit.py" \
      --root "${result_root}" --data-root "${DATA_ROOT}" \
      --output "${result_root}/stage1_summary.json" \
      >"${result_root}/stage1_evaluate.log" 2>&1
  printf '%s\n' \
    'dense_artifacts=retained_remotely_for_frozen_stage3_only' \
    'cleanup_due=immediately_after_stage3_compact_artifacts_are_verified' \
    >"${result_root}/stage1_dense_retention_exception.txt"
  (
    cd "${result_root}"
    sha256sum preflight.txt stage1_summary.json stage1_evaluate.log \
      stage1_dense_retention_exception.txt >stage1_compact_sha256.txt
  )
  printf 'run_name=%s\nstatus=complete\ncompleted_utc=%s\n' \
    "${RUN_NAME}" "$(date -u +%FT%TZ)" >"${result_root}/stage1_complete.txt"
}

run_stage2_object() {
  local gpu="$1" object_name="$2"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage2"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${snapshot}:${snapshot}/src:/workspace" \
    python3 "${snapshot}/scripts/run_flow_tte_support_bank_routing.py" \
      --data-root "${DATA_ROOT}" \
      --support-json "${snapshot}/${SUPPORT_REL}" \
      --output-root "${result_root}" \
      --project-root /workspace --fsad-root "${snapshot}" \
      --objects "${object_name}" --device cuda \
      >"${result_root}/logs/gpu${gpu}_${object_name}.log" 2>&1
  PYTHONPATH="${snapshot}:${snapshot}/src" \
    python3 "${snapshot}/scripts/evaluate_flow_tte_support_bank_routing.py" \
      --root "${result_root}" --data-root "${DATA_ROOT}" \
      --objects "${object_name}" \
      --output "${result_root}/objects/${object_name}/stage2_metrics.json" \
      >"${result_root}/logs/gpu${gpu}_${object_name}_evaluate.log" 2>&1
  find "${result_root}/objects/${object_name}" -type d -name anomaly_maps \
    -prune -exec rm -rf -- {} +
  if find "${result_root}/objects/${object_name}" -type d -name anomaly_maps \
      -print -quit | grep -q .; then
    printf 'Stage 2 object dense map cleanup failed: %s\n' "${object_name}" >&2
    return 1
  fi
  printf 'gpu=%s object=%s stage2=complete\n' "${gpu}" "${object_name}" \
    >"${result_root}/objects/${object_name}/stage2_complete.txt"
}

run_stage2_group() {
  local gpu="$1"
  shift
  local object_name
  for object_name in "$@"; do
    run_stage2_object "${gpu}" "${object_name}"
  done
  printf 'gpu=%s status=complete objects=%s\n' "${gpu}" "$*" \
    >"${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage2/gpu${gpu}_complete.txt"
}

internal_stage2() {
  require_run_name
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage2"
  record_failure() {
    local exit_code="$?"
    if [[ "${exit_code}" -ne 0 && ! -f "${result_root}/complete.txt" ]]; then
      printf 'stage2 controller failed with exit_code=%s\n' "${exit_code}" \
        >"${result_root}/failed.txt"
    fi
  }
  trap record_failure EXIT
  exec 9>"${REMOTE_SNAPSHOT_ROOT}/.default_shift_sequence_gpus_0_3.lock"
  flock -n 9 || { printf 'cooperative GPU lock is held\n' \
    >"${result_root}/failed.txt"; return 75; }
  test -f "${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage1_complete.txt"
  printf 'controller_pid=%s\nstarted_utc=%s\n' "$$" "$(date -u +%FT%TZ)" \
    >"${result_root}/started.txt"
  read -r -a group0 <<<"${OBJECT_GROUP_0}"
  read -r -a group1 <<<"${OBJECT_GROUP_1}"
  read -r -a group2 <<<"${OBJECT_GROUP_2}"
  read -r -a group3 <<<"${OBJECT_GROUP_3}"
  status=0
  run_stage2_group 0 "${group0[@]}" & p0=$!
  run_stage2_group 1 "${group1[@]}" & p1=$!
  for pid in "${p0}" "${p1}"; do
    wait "${pid}" || status=1
  done
  run_stage2_group 2 "${group2[@]}" & p2=$!
  run_stage2_group 3 "${group3[@]}" & p3=$!
  for pid in "${p2}" "${p3}"; do
    wait "${pid}" || status=1
  done
  if [[ "${status}" -ne 0 ]]; then
    printf 'one or more Stage 2 GPU groups failed\n' >"${result_root}/failed.txt"
    return 1
  fi
  PYTHONPATH="${snapshot}:${snapshot}/src" \
    python3 "${snapshot}/scripts/evaluate_flow_tte_support_bank_routing.py" \
      --root "${result_root}" --data-root "${DATA_ROOT}" \
      --from-object-summaries \
      --output "${result_root}/summary.json" \
      >"${result_root}/evaluate.log" 2>&1
  find "${result_root}" -type d -name anomaly_maps -prune -exec rm -rf -- {} +
  if find "${result_root}" -type d -name anomaly_maps -print -quit | grep -q .; then
    printf 'Stage 2 dense map cleanup failed\n' >"${result_root}/failed.txt"
    return 1
  fi
  printf 'anomaly_maps=removed_after_verified_summary\n' >"${result_root}/cleanup_evidence.txt"
  (
    cd "${result_root}"
    sha256sum preflight.txt summary.json evaluate.log cleanup_evidence.txt \
      objects/*/routing.json objects/*/runner_manifest.json \
      objects/*/stage2_metrics.json \
      >compact_sha256.txt
  )
  printf 'run_name=%s\nstage=2\nstatus=complete\ncompleted_utc=%s\n' \
    "${RUN_NAME}" "$(date -u +%FT%TZ)" >"${result_root}/complete.txt"
}

internal_stage3() {
  require_run_name
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo"
  local sequence_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local result_root="${sequence_root}/stage3"
  record_failure() {
    local exit_code="$?"
    if [[ "${exit_code}" -ne 0 && ! -f "${result_root}/complete.txt" ]]; then
      printf 'stage3 controller failed with exit_code=%s\n' "${exit_code}" \
        >"${result_root}/failed.txt"
    fi
  }
  trap record_failure EXIT
  test -f "${sequence_root}/stage1_complete.txt"
  test -f "${sequence_root}/stage2/complete.txt"
  printf 'controller_pid=%s\nstarted_utc=%s\n' "$$" "$(date -u +%FT%TZ)" \
    >"${result_root}/started.txt"
  PYTHONPATH="${snapshot}:${snapshot}/src" \
    python3 "${snapshot}/scripts/evaluate_flow_tte_guided_fallback.py" \
      --root "${sequence_root}" --data-root "${DATA_ROOT}" \
      --output "${result_root}/summary.json" \
      >"${result_root}/evaluate.log" 2>&1
  find "${sequence_root}/objects" -type d -name anomaly_maps \
    -prune -exec rm -rf -- {} +
  if find "${sequence_root}/objects" -type d -name anomaly_maps \
      -print -quit | grep -q .; then
    printf 'Stage 1 dense map cleanup after Stage 3 failed\n' >"${result_root}/failed.txt"
    return 1
  fi
  printf '%s\n' \
    'stage1_anomaly_maps=removed_after_stage3_summary_verified' \
    'stage2_anomaly_maps=removed_by_stage2_controller' \
    >"${result_root}/cleanup_evidence.txt"
  (
    cd "${result_root}"
    sha256sum summary.json evaluate.log cleanup_evidence.txt >compact_sha256.txt
  )
  printf 'run_name=%s\nstage=3\nstatus=complete\ncompleted_utc=%s\n' \
    "${RUN_NAME}" "$(date -u +%FT%TZ)" >"${result_root}/complete.txt"
}

run_stage4_object() {
  local gpu="$1" object_name="$2"
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo"
  local result_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage4"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${snapshot}:${snapshot}/src:/workspace" \
    python3 "${snapshot}/scripts/run_flow_tte_local_soft_nuisance.py" \
      --data-root "${DATA_ROOT}" \
      --support-json "${snapshot}/${SUPPORT_REL}" \
      --output-root "${result_root}" \
      --project-root /workspace --fsad-root "${snapshot}" \
      --objects "${object_name}" --device cuda \
      >"${result_root}/logs/gpu${gpu}_${object_name}.log" 2>&1
  printf 'gpu=%s object=%s stage4=complete\n' "${gpu}" "${object_name}" \
    >"${result_root}/objects/${object_name}/stage4_complete.txt"
}

run_stage4_group() {
  local gpu="$1"
  shift
  local object_name
  for object_name in "$@"; do
    run_stage4_object "${gpu}" "${object_name}"
  done
  printf 'gpu=%s status=complete objects=%s\n' "${gpu}" "$*" \
    >"${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage4/gpu${gpu}_complete.txt"
}

internal_stage4() {
  require_run_name
  local snapshot="${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2/repo"
  local sequence_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  local result_root="${sequence_root}/stage4"
  record_failure() {
    local exit_code="$?"
    if [[ "${exit_code}" -ne 0 && ! -f "${result_root}/complete.txt" ]]; then
      printf 'stage4 controller failed with exit_code=%s\n' "${exit_code}" \
        >"${result_root}/failed.txt"
    fi
  }
  trap record_failure EXIT
  exec 9>"${REMOTE_SNAPSHOT_ROOT}/.default_shift_sequence_gpus_0_3.lock"
  flock -n 9 || { printf 'cooperative GPU lock is held\n' \
    >"${result_root}/failed.txt"; return 75; }
  test -f "${sequence_root}/stage3/complete.txt"
  printf 'controller_pid=%s\nstarted_utc=%s\n' "$$" "$(date -u +%FT%TZ)" \
    >"${result_root}/started.txt"
  read -r -a group0 <<<"${OBJECT_GROUP_0}"
  read -r -a group1 <<<"${OBJECT_GROUP_1}"
  read -r -a group2 <<<"${OBJECT_GROUP_2}"
  read -r -a group3 <<<"${OBJECT_GROUP_3}"
  run_stage4_group 0 "${group0[@]}" & p0=$!
  run_stage4_group 1 "${group1[@]}" & p1=$!
  run_stage4_group 2 "${group2[@]}" & p2=$!
  run_stage4_group 3 "${group3[@]}" & p3=$!
  status=0
  for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
    wait "${pid}" || status=1
  done
  if [[ "${status}" -ne 0 ]]; then
    printf 'one or more Stage 4 GPU groups failed\n' >"${result_root}/failed.txt"
    return 1
  fi
  PYTHONPATH="${snapshot}:${snapshot}/src" \
    python3 "${snapshot}/scripts/evaluate_flow_tte_local_soft_nuisance.py" \
      --root "${result_root}" --data-root "${DATA_ROOT}" \
      --output "${result_root}/summary.json" \
      >"${result_root}/evaluate.log" 2>&1
  find "${result_root}" -type d -name anomaly_maps -prune -exec rm -rf -- {} +
  if find "${result_root}" -type d -name anomaly_maps -print -quit | grep -q .; then
    printf 'Stage 4 dense map cleanup failed\n' >"${result_root}/failed.txt"
    return 1
  fi
  printf 'anomaly_maps=removed_after_verified_summary\n' >"${result_root}/cleanup_evidence.txt"
  (
    cd "${result_root}"
    sha256sum preflight.txt summary.json evaluate.log cleanup_evidence.txt \
      objects/*/manifest.json >compact_sha256.txt
  )
  printf 'run_name=%s\nstage=4\nstatus=complete\ncompleted_utc=%s\n' \
    "${RUN_NAME}" "$(date -u +%FT%TZ)" >"${result_root}/complete.txt"
}

status_remote() {
  require_run_name
  assert_fixed_container
  ssh_remote "${REMOTE_HOST}" "docker exec '${FIXED_CONTAINER}' bash -lc '
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    snapshot=${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}
    printf \"run_name=%s\\n\" ${RUN_NAME}
    test -e \"\${root}\" && echo result_root_exists=true || echo result_root_exists=false
    test -e \"\${snapshot}\" && echo snapshot_exists=true || echo snapshot_exists=false
    if test -e \"\${root}\"; then
      printf \"result_filesystem_free_gib=%s\\n\" \\
        \"\$(df -BG --output=avail \"\${root}\" | tail -n 1 | tr -dc '0-9')\"
    fi
    test -f \"\${root}/stage1_complete.txt\" && echo stage1=complete \\
      || { test -f \"\${root}/stage1_failed.txt\" && echo stage1=failed \\
        || echo stage1=running_or_pending; }
    test -f \"\${root}/stage1_failed.txt\" && cat \"\${root}/stage1_failed.txt\" || true
    printf \"objects_complete=%s\\n\" \\
      \"\$(find \"\${root}/objects\" -name stage1_complete.txt 2>/dev/null | wc -l)\"
    printf \"gpu_groups_complete=%s\\n\" \\
      \"\$(find \"\${root}\" -maxdepth 1 -name 'gpu*_stage1_complete.txt' 2>/dev/null | wc -l)\"
    if test -s \"\${root}/controller_stage1.pid\"; then
      ps -o pid=,stat=,etime=,args= -p \"\$(cat \"\${root}/controller_stage1.pid\")\" || true
    fi
    ps -eo pid=,ppid=,stat=,etime=,%cpu=,%mem=,rss=,args= \
      | grep -E \"run_flow_tte_mvtec_ad2.py|guided_refinement.py|default_shift_audit.py\" \
      | grep -v grep || true
    for log in \"\${root}\"/logs/*.log; do
      test -e \"\${log}\" || continue
      printf \"log=%s bytes=%s\\n\" \"\$(basename \"\${log}\")\" \"\$(wc -c <\"\${log}\")\"
      tail -n 1 \"\${log}\" | sed \"s/^/  last=/\"
    done
    if test -d \"\${root}/stage2\"; then
      test -f \"\${root}/stage2/complete.txt\" && echo stage2=complete \\
        || { test -f \"\${root}/stage2/failed.txt\" && echo stage2=failed \\
          || echo stage2=running_or_pending; }
      printf \"stage2_objects_complete=%s\\n\" \\
        \"\$(find \"\${root}/stage2/objects\" -name stage2_complete.txt 2>/dev/null | wc -l)\"
      test -f \"\${root}/stage2/failed.txt\" && cat \"\${root}/stage2/failed.txt\" || true
      ps -eo pid=,ppid=,stat=,etime=,%cpu=,%mem=,rss=,args= \\
        | grep -E \"run_flow_tte_support_bank_routing.py|evaluate_flow_tte_support_bank_routing.py\" \\
        | grep -v grep || true
      for log in \"\${root}/stage2/logs\"/*.log; do
        test -e \"\${log}\" || continue
        printf \"stage2_log=%s bytes=%s\\n\" \\
          \"\$(basename \"\${log}\")\" \"\$(wc -c <\"\${log}\")\"
        tail -n 1 \"\${log}\" | sed \"s/^/  last=/\"
      done
    fi
    if test -d \"\${root}/stage3\"; then
      test -f \"\${root}/stage3/complete.txt\" && echo stage3=complete \\
        || { test -f \"\${root}/stage3/failed.txt\" && echo stage3=failed \\
          || echo stage3=running_or_pending; }
      test -f \"\${root}/stage3/failed.txt\" && cat \"\${root}/stage3/failed.txt\" || true
      ps -eo pid=,ppid=,stat=,etime=,%cpu=,%mem=,rss=,args= \\
        | grep -E \"evaluate_flow_tte_guided_fallback.py\" \\
        | grep -v grep || true
      if test -e \"\${root}/stage3/evaluate.log\"; then
        printf \"stage3_log_bytes=%s\\n\" \"\$(wc -c <\"\${root}/stage3/evaluate.log\")\"
        tail -n 1 \"\${root}/stage3/evaluate.log\" | sed \"s/^/  last=/\"
      fi
    fi
    if test -d \"\${root}/stage4\"; then
      test -f \"\${root}/stage4/complete.txt\" && echo stage4=complete \\
        || { test -f \"\${root}/stage4/failed.txt\" && echo stage4=failed \\
          || echo stage4=running_or_pending; }
      printf \"stage4_objects_complete=%s\\n\" \\
        \"\$(find \"\${root}/stage4/objects\" -name stage4_complete.txt 2>/dev/null | wc -l)\"
      test -f \"\${root}/stage4/failed.txt\" && cat \"\${root}/stage4/failed.txt\" || true
      ps -eo pid=,ppid=,stat=,etime=,%cpu=,%mem=,rss=,args= \\
        | grep -E \"run_flow_tte_local_soft_nuisance.py|evaluate_flow_tte_local_soft_nuisance.py\" \\
        | grep -v grep || true
      for log in \"\${root}/stage4/logs\"/*.log; do
        test -e \"\${log}\" || continue
        printf \"stage4_log=%s bytes=%s\\n\" \\
          \"\$(basename \"\${log}\")\" \"\$(wc -c <\"\${log}\")\"
        tail -n 1 \"\${log}\" | sed \"s/^/  last=/\"
      done
      if test -e \"\${root}/stage4/evaluate.log\"; then
        printf \"stage4_evaluate_log_bytes=%s\\n\" \\
          \"\$(wc -c <\"\${root}/stage4/evaluate.log\")\"
        tail -n 1 \"\${root}/stage4/evaluate.log\" | sed \"s/^/  last=/\"
      fi
    fi
  '"
  ssh_remote "${REMOTE_HOST}" \
    "nvidia-smi -i 0,1,2,3 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader"
}

pull_stage1() {
  require_run_name
  assert_fixed_container
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}/stage1"
  local partial="${destination}.partial.$$"
  [[ ! -e "${destination}" ]] || die "pull destination exists: ${destination}"
  mkdir -p "$(dirname "${destination}")" "${partial}"
  trap 'rm -rf -- "'"${partial}"'"' EXIT
  ssh_remote "${REMOTE_HOST}" \
    "docker exec '${FIXED_CONTAINER}' tar \
      --exclude='*/anomaly_maps' -C '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' -cf - ." \
    | tar -C "${partial}" -xf -
  test -f "${partial}/stage1_complete.txt"
  (cd "${partial}" && sha256sum -c stage1_compact_sha256.txt)
  if find "${partial}" -type d -name anomaly_maps -print -quit | grep -q .; then
    die "compact pull contains anomaly maps"
  fi
  mv "${partial}" "${destination}"
  trap - EXIT
  printf 'PULLED stage1 destination=%s\n' "${destination}"
}

pull_stage2() {
  require_run_name
  assert_fixed_container
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}/stage2"
  local partial="${destination}.partial.$$"
  [[ ! -e "${destination}" ]] || die "pull destination exists: ${destination}"
  mkdir -p "$(dirname "${destination}")" "${partial}"
  trap 'rm -rf -- "'"${partial}"'"' EXIT
  ssh_remote "${REMOTE_HOST}" \
    "docker exec '${FIXED_CONTAINER}' tar \
      -C '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage2' -cf - ." \
    | tar -C "${partial}" -xf -
  test -f "${partial}/complete.txt"
  (cd "${partial}" && sha256sum -c compact_sha256.txt)
  if find "${partial}" -type d -name anomaly_maps -print -quit | grep -q .; then
    die "Stage 2 pull contains anomaly maps"
  fi
  mv "${partial}" "${destination}"
  trap - EXIT
  printf 'PULLED stage2 destination=%s\n' "${destination}"
}

pull_stage3() {
  require_run_name
  assert_fixed_container
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}/stage3"
  local partial="${destination}.partial.$$"
  [[ ! -e "${destination}" ]] || die "pull destination exists: ${destination}"
  mkdir -p "$(dirname "${destination}")" "${partial}"
  trap 'rm -rf -- "'"${partial}"'"' EXIT
  ssh_remote "${REMOTE_HOST}" \
    "docker exec '${FIXED_CONTAINER}' tar \
      -C '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage3' -cf - ." \
    | tar -C "${partial}" -xf -
  test -f "${partial}/complete.txt"
  (cd "${partial}" && sha256sum -c compact_sha256.txt)
  mv "${partial}" "${destination}"
  trap - EXIT
  printf 'PULLED stage3 destination=%s\n' "${destination}"
}

pull_stage4() {
  require_run_name
  assert_fixed_container
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}/stage4"
  local partial="${destination}.partial.$$"
  [[ ! -e "${destination}" ]] || die "pull destination exists: ${destination}"
  mkdir -p "$(dirname "${destination}")" "${partial}"
  trap 'rm -rf -- "'"${partial}"'"' EXIT
  ssh_remote "${REMOTE_HOST}" \
    "docker exec '${FIXED_CONTAINER}' tar \
      -C '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage4' -cf - ." \
    | tar -C "${partial}" -xf -
  test -f "${partial}/complete.txt"
  (cd "${partial}" && sha256sum -c compact_sha256.txt)
  if find "${partial}" -type d -name anomaly_maps -print -quit | grep -q .; then
    die "Stage 4 pull contains anomaly maps"
  fi
  mv "${partial}" "${destination}"
  trap - EXIT
  printf 'PULLED stage4 destination=%s\n' "${destination}"
}

patch_stage1_evaluator() {
  require_run_name
  assert_fixed_container
  local source="${ROOT_DIR}/scripts/evaluate_flow_tte_default_shift_audit.py"
  local optimized_hash
  optimized_hash="$(sha256sum "${source}" | awk '{print $1}')"
  ssh_remote "${REMOTE_HOST}" \
    "docker exec -i '${FIXED_CONTAINER}' bash -lc '
      set -euo pipefail
      snapshot=${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo
      result_root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
      destination=\"\${snapshot}/scripts/evaluate_flow_tte_default_shift_audit.py\"
      temporary=\"\${destination}.optimized.tmp\"
      test -f \"\${result_root}/stage1_started.txt\"
      test ! -e \"\${result_root}/stage1_evaluate.log\"
      current=\$(sha256sum \"\${destination}\" | awk \"{print \\\$1}\")
      test \"\${current}\" = \"${STAGE1_EVALUATOR_PREOPT_SHA256}\" \\
        || test \"\${current}\" = \"${STAGE1_EVALUATOR_OPT1_SHA256}\" \\
        || test \"\${current}\" = \"${STAGE1_EVALUATOR_OPT2_SHA256}\"
      tee \"\${temporary}\" >/dev/null
      actual=\$(sha256sum \"\${temporary}\" | awk \"{print \\\$1}\")
      test \"\${actual}\" = \"${optimized_hash}\"
      chmod +x \"\${temporary}\"
      mv \"\${temporary}\" \"\${destination}\"
      printf \"change=pre_metric_evaluator_correction\\nold_sha256=%s\\nnew_sha256=%s\\ngate_semantics=unchanged\\n---\\n\" \\
        \"\${current}\" \"\${actual}\" >>\"\${result_root}/evaluator_optimization_amendment.txt\"
      printf \"PATCHED stage1_evaluator old=%s new=%s\\n\" \"\${current}\" \"\${actual}\"
    '" <"${source}"
}

patch_stage1_embedded_evaluation() {
  require_run_name
  assert_fixed_container
  local runner="${ROOT_DIR}/scripts/run_flow_tte_mvtec_ad2.py"
  local guide="${ROOT_DIR}/scripts/run_flow_tte_mvtecad2_guided_refinement.py"
  local runner_hash guide_hash archive
  runner_hash="$(sha256sum "${runner}" | awk '{print $1}')"
  guide_hash="$(sha256sum "${guide}" | awk '{print $1}')"
  archive="$(mktemp "${TMPDIR:-/tmp}/stage1-skip-eval.XXXXXX.tar")"
  trap 'rm -f -- "'"${archive}"'"' EXIT
  tar -C "${ROOT_DIR}" -cf "${archive}" \
    scripts/run_flow_tte_mvtec_ad2.py \
    scripts/run_flow_tte_mvtecad2_guided_refinement.py
  ssh_remote "${REMOTE_HOST}" \
    "docker exec -i '${FIXED_CONTAINER}' tar \
      -C '${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo' -xf -" <"${archive}"
  rm -f -- "${archive}"
  trap - EXIT
  ssh_remote "${REMOTE_HOST}" "docker exec -i '${FIXED_CONTAINER}' bash -s -- \
    '${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}/repo' \
    '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' \
    '${runner_hash}' '${guide_hash}'" <<'REMOTE_PATCH_VERIFY'
set -euo pipefail
snapshot="$1"
result_root="$2"
runner_hash="$3"
guide_hash="$4"
test -f "${result_root}/stage1_started.txt"
test ! -e "${result_root}/stage1_evaluate.log"
test "$(sha256sum "${snapshot}/scripts/run_flow_tte_mvtec_ad2.py" | awk '{print $1}')" \
  = "${runner_hash}"
test "$(sha256sum "${snapshot}/scripts/run_flow_tte_mvtecad2_guided_refinement.py" | awk '{print $1}')" \
  = "${guide_hash}"
python3 -m py_compile \
  "${snapshot}/scripts/run_flow_tte_mvtec_ad2.py" \
  "${snapshot}/scripts/run_flow_tte_mvtecad2_guided_refinement.py"
touch "${snapshot}/scripts/.skip_embedded_public_evaluation"
printf '%s\n' \
  'change=skip_redundant_embedded_public_evaluation_after_map_write' \
  "runner_new_sha256=${runner_hash}" \
  "guide_new_sha256=${guide_hash}" \
  'previous_hashes=recorded_in_preflight.txt' \
  'map_generation=unchanged' \
  'final_metrics=frozen_stage1_external_audit' \
  >>"${result_root}/embedded_evaluation_amendment.txt"
printf 'PATCHED stage1 embedded evaluation runner=%s guide=%s\n' \
  "${runner_hash}" "${guide_hash}"
REMOTE_PATCH_VERIFY
}

cleanup_aborted_start() {
  require_run_name
  assert_fixed_container
  ssh_remote "${REMOTE_HOST}" "docker exec -i '${FIXED_CONTAINER}' bash -s -- \
    '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' \
    '${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}'" <<'REMOTE_ABORTED_CLEANUP'
set -euo pipefail
result_root="$1"
snapshot_root="$2"
test ! -f "${result_root}/stage1_started.txt"
test ! -f "${result_root}/stage1_complete.txt"
rm -rf -- "${result_root}" "${snapshot_root}"
test ! -e "${result_root}"
test ! -e "${snapshot_root}"
printf 'aborted_start_cleanup=complete\n'
REMOTE_ABORTED_CLEANUP
}

cleanup_aborted_stage2() {
  require_run_name
  assert_fixed_container
  ssh_remote "${REMOTE_HOST}" "docker exec -i '${FIXED_CONTAINER}' bash -s -- \
    '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage2' \
    '${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2'" <<'REMOTE_STAGE2_ABORTED_CLEANUP'
set -euo pipefail
result_root="$1"
snapshot_root="$2"
test ! -f "${result_root}/started.txt"
test ! -f "${result_root}/complete.txt"
rm -rf -- "${result_root}" "${snapshot_root}"
test ! -e "${result_root}"
test ! -e "${snapshot_root}"
printf 'aborted_stage2_cleanup=complete\n'
REMOTE_STAGE2_ABORTED_CLEANUP
}

cleanup_failed_stage2() {
  require_run_name
  assert_fixed_container
  ssh_remote "${REMOTE_HOST}" "docker exec -i '${FIXED_CONTAINER}' bash -s -- \
    '${REMOTE_RESULTS_ROOT}/${RUN_NAME}/stage2' \
    '${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2' \
    '${RUN_NAME}'" <<'REMOTE_STAGE2_FAILED_CLEANUP'
set -euo pipefail
result_root="$1"
snapshot_root="$2"
run_name="$3"
test -f "${result_root}/failed.txt"
test ! -f "${result_root}/complete.txt"
if pgrep -af "${run_name}_stage2/repo/scripts/(run|evaluate)_flow_tte_support_bank_routing.py"; then
  echo "refusing cleanup while Stage 2 processes remain" >&2
  exit 78
fi
rm -rf -- "${result_root}" "${snapshot_root}"
test ! -e "${result_root}"
test ! -e "${snapshot_root}"
printf 'failed_stage2_cleanup=complete\n'
REMOTE_STAGE2_FAILED_CLEANUP
}

cleanup_complete_sequence() {
  require_run_name
  assert_fixed_container
  ssh_remote "${REMOTE_HOST}" "docker exec -i '${FIXED_CONTAINER}' bash -s -- \
    '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' \
    '${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}' \
    '${REMOTE_SNAPSHOT_ROOT}/${RUN_NAME}_stage2'" <<'REMOTE_COMPLETE_CLEANUP'
set -euo pipefail
result_root="$1"
stage1_snapshot="$2"
stage2_snapshot="$3"
test -f "${result_root}/stage1_complete.txt"
test -f "${result_root}/stage2/complete.txt"
test -f "${result_root}/stage3/complete.txt"
test -f "${result_root}/stage4/complete.txt"
if find "${result_root}" -type d -name anomaly_maps -print -quit | grep -q .; then
  echo "refusing final cleanup while anomaly maps remain" >&2
  exit 77
fi
rm -rf -- "${stage1_snapshot}" "${stage2_snapshot}"
test ! -e "${stage1_snapshot}"
test ! -e "${stage2_snapshot}"
printf '%s\n' \
  'cleanup=complete' \
  "stage1_snapshot_removed=${stage1_snapshot}" \
  "stage2_snapshot_removed=${stage2_snapshot}" \
  'dense_artifact_directories=0' \
  'compact_results_retained=true' \
  >"${result_root}/final_cleanup.txt"
printf 'COMPLETE_SEQUENCE_CLEANUP run_root=%s\n' "${result_root}"
REMOTE_COMPLETE_CLEANUP
}

pull_cleanup_evidence() {
  require_run_name
  assert_fixed_container
  local destination="${LOCAL_PULL_ROOT}/${RUN_NAME}/cleanup"
  local partial="${destination}.partial.$$"
  [[ ! -e "${destination}" ]] || die "pull destination exists: ${destination}"
  mkdir -p "$(dirname "${destination}")" "${partial}"
  trap 'rm -rf -- "'"${partial}"'"' EXIT
  ssh_remote "${REMOTE_HOST}" \
    "docker exec '${FIXED_CONTAINER}' tar \
      -C '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' -cf - final_cleanup.txt" \
    | tar -C "${partial}" -xf -
  test -f "${partial}/final_cleanup.txt"
  mv "${partial}" "${destination}"
  trap - EXIT
  printf 'PULLED cleanup destination=%s\n' "${destination}"
}

case "${MODE}" in
  start-stage1)
    if [[ -z "${RUN_NAME:-}" ]]; then
      RUN_NAME="default_shift_sequence_$(date -u +%Y%m%dT%H%M%SZ)_$$"
    fi
    load_remote_preset
    start_stage1
    ;;
  start-stage2)
    load_remote_preset
    start_stage2
    ;;
  start-stage3)
    load_remote_preset
    start_stage3
    ;;
  start-stage4)
    load_remote_preset
    start_stage4
    ;;
  status)
    load_remote_preset
    status_remote
    ;;
  pull-stage1)
    load_remote_preset
    pull_stage1
    ;;
  pull-stage2)
    load_remote_preset
    pull_stage2
    ;;
  pull-stage3)
    load_remote_preset
    pull_stage3
    ;;
  pull-stage4)
    load_remote_preset
    pull_stage4
    ;;
  patch-stage1-evaluator)
    load_remote_preset
    patch_stage1_evaluator
    ;;
  patch-stage1-embedded-evaluation)
    load_remote_preset
    patch_stage1_embedded_evaluation
    ;;
  cleanup-complete)
    load_remote_preset
    cleanup_complete_sequence
    ;;
  pull-cleanup)
    load_remote_preset
    pull_cleanup_evidence
    ;;
  cleanup-aborted)
    load_remote_preset
    cleanup_aborted_start
    ;;
  cleanup-aborted-stage2)
    load_remote_preset
    cleanup_aborted_stage2
    ;;
  cleanup-failed-stage2)
    load_remote_preset
    cleanup_failed_stage2
    ;;
  internal-stage1)
    internal_stage1
    ;;
  internal-stage2)
    internal_stage2
    ;;
  internal-stage3)
    internal_stage3
    ;;
  internal-stage4)
    internal_stage4
    ;;
  *)
    die "usage: RUN_NAME=default_shift_sequence_<id> $0 {start-stage1|start-stage2|start-stage3|start-stage4|status|pull-stage1|pull-stage2|pull-stage3|pull-stage4|patch-stage1-evaluator|patch-stage1-embedded-evaluation|cleanup-aborted|cleanup-aborted-stage2|cleanup-failed-stage2|cleanup-complete|pull-cleanup}"
    ;;
esac
