#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-plan}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-flowtte_basic_dinov2_hparam_factorial_20260713_v1}"
EXPERIMENT_KIND="${EXPERIMENT_KIND:-hparam}"
SELECTED_DEPTH="${SELECTED_DEPTH:-2}"
SELECTED_WIDTH="${SELECTED_WIDTH:-1}"
SELECTED_EPOCHS="${SELECTED_EPOCHS:-3}"
SELECTED_LR="${SELECTED_LR:-2e-4}"
SELECTED_LOGDET="${SELECTED_LOGDET:-1e-3}"
SELECTED_BRIGHTNESS="${SELECTED_BRIGHTNESS:-1.0,1.0}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/fsad_tta}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/workspace/results_remote}"
SUPPORT_JSON_REL="skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json"
ASKPASS_SCRIPT="${ROOT_DIR}/scripts/ssh_askpass_env.sh"
GPU_IDLE_MEMORY_MIB="${GPU_IDLE_MEMORY_MIB:-512}"
GPU_IDLE_POLL_SECONDS="${GPU_IDLE_POLL_SECONDS:-30}"

OBJECTS=(can fabric fruit_jelly rice vial wallplugs walnuts sheet_metal)

print_plan() {
  cat <<'EOF'
host	gpu	queue
dsba3	0	cap_d1_w1,reg_ld2e2_b08
dsba3	1	cap_d1_w2,reg_ld2e2_b10
dsba3	2	cap_d2_w1,reg_ld1e3_b08
dsba3	3	cap_d2_w2
dsba5	0	cap_d4_w1
dsba5	1	cap_d4_w2

config	depth	width	lambda_logdet	brightness
cap_d1_w1	1	1	1e-3	1.0,1.0
cap_d1_w2	1	2	1e-3	1.0,1.0
cap_d2_w1	2	1	1e-3	1.0,1.0
cap_d2_w2	2	2	1e-3	1.0,1.0
cap_d4_w1	4	1	1e-3	1.0,1.0
cap_d4_w2	4	2	1e-3	1.0,1.0
reg_ld1e3_b08	2	1	1e-3	0.8,1.2
reg_ld2e2_b10	2	1	2e-2	1.0,1.0
reg_ld2e2_b08	2	1	2e-2	0.8,1.2
EOF
}

load_preset() {
  local preset="$1"
  # shellcheck disable=SC1090
  source "${preset}"
}

ssh_remote() {
  local args=(-p "${REMOTE_PORT}")
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

sync_repo() {
  tar --exclude=.git --exclude=.omx --exclude=PaperWorks --exclude=configs/remote \
    --exclude=results --exclude='*.tgz' -C "${ROOT_DIR}" -cf - . \
    | ssh_remote "${REMOTE_HOST}" \
      "docker exec -i '${CONTAINER_NAME}' bash -lc '
        set -e
        mkdir -p ${REMOTE_REPO} /workspace/src
        tar -C ${REMOTE_REPO} -xf -
        cp ${REMOTE_REPO}/src/backbones.py /workspace/src/backbones.py
        cp ${REMOTE_REPO}/src/dinov2_loader.py /workspace/src/dinov2_loader.py
      '"
}

host_data_root() {
  case "$1" in
    dsba3) printf '%s\n' /home/hunim/Volume/DATA/mvtec_ad_2 ;;
    dsba5) printf '%s\n' /workspace/data/mvtec_ad_2 ;;
    *) return 2 ;;
  esac
}

host_gpu_count() {
  case "$1" in
    dsba3) printf '%s\n' 4 ;;
    dsba5) printf '%s\n' 2 ;;
    *) return 2 ;;
  esac
}

preflight_host() {
  local tag="$1"
  local data_root="$2"
  local expected_gpus="$3"
  ssh_remote "${REMOTE_HOST}" "docker exec '${CONTAINER_NAME}' bash -lc '
    set -euo pipefail
    test -d \"${data_root}\"
    test -f \"${REMOTE_REPO}/${SUPPORT_JSON_REL}\"
    test \"\$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)\" -eq \"${expected_gpus}\"
    TORCH_HOME=/workspace/.cache/torch FMAD_DINOV2_OFFLINE=1 \
      PYTHONPATH=${REMOTE_REPO} python3 ${REMOTE_REPO}/scripts/preflight_dinov2_cache.py \
        --model dinov2_vitl14 \
      >/tmp/${RUN_NAME}_${tag}_dinov2_preflight.log
    for object in ${OBJECTS[*]}; do
      test -n \"\$(find \"${data_root}/\${object}/train/good\" -maxdepth 1 -type f -name \"*.png\" -print -quit)\"
      test -n \"\$(find \"${data_root}/\${object}/test_public/good\" -maxdepth 1 -type f -name \"*.png\" -print -quit)\"
      test -n \"\$(find \"${data_root}/\${object}/test_public/bad\" -maxdepth 1 -type f -name \"*.png\" -print -quit)\"
    done
    echo PREFLIGHT_OK host=${tag} gpu_count=${expected_gpus} data_root=${data_root}
  '"
}

start_host() {
  local tag="$1"
  local preset="${ROOT_DIR}/configs/remote/${tag}.env"
  local data_root expected_gpus
  load_preset "${preset}"
  data_root="$(host_data_root "${tag}")"
  expected_gpus="$(host_gpu_count "${tag}")"
  sync_repo
  preflight_host "${tag}" "${data_root}" "${expected_gpus}"
  ssh_remote "${REMOTE_HOST}" "docker exec '${CONTAINER_NAME}' bash -lc '
    set -euo pipefail
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    mkdir -p \"\${root}\"
    if test -f \"\${root}/${tag}_complete.txt\"; then
      echo ALREADY_COMPLETE host=${tag}
      exit 0
    fi
    if test -s \"\${root}/${tag}_controller.pid\" \\
      && kill -0 \"\$(cat \"\${root}/${tag}_controller.pid\")\" 2>/dev/null \\
      && ! ps -o stat= -p \"\$(cat \"\${root}/${tag}_controller.pid\")\" | grep -q \"^[[:space:]]*Z\"; then
      echo ALREADY_RUNNING host=${tag} pid=\$(cat \"\${root}/${tag}_controller.pid\")
      exit 0
    fi
    nohup env RUN_NAME=${RUN_NAME} REMOTE_REPO=${REMOTE_REPO} \
      EXPERIMENT_KIND=${EXPERIMENT_KIND} \
      SELECTED_DEPTH=${SELECTED_DEPTH} SELECTED_WIDTH=${SELECTED_WIDTH} \
      SELECTED_EPOCHS=${SELECTED_EPOCHS} SELECTED_LR=${SELECTED_LR} \
      SELECTED_LOGDET=${SELECTED_LOGDET} SELECTED_BRIGHTNESS=${SELECTED_BRIGHTNESS} \
      REMOTE_RESULTS_ROOT=${REMOTE_RESULTS_ROOT} DATA_ROOT=${data_root} \
      GPU_IDLE_MEMORY_MIB=${GPU_IDLE_MEMORY_MIB} \
      GPU_IDLE_POLL_SECONDS=${GPU_IDLE_POLL_SECONDS} \
      bash ${REMOTE_REPO}/scripts/run_flow_tte_basic_hparam_parallel_remote.sh \
        internal-controller ${tag} \
      >\"\${root}/${tag}_controller.log\" 2>&1 </dev/null &
    echo \$! >\"\${root}/${tag}_controller.pid\"
    echo STARTED host=${tag} pid=\$! log=\${root}/${tag}_controller.log
  '"
}

config_values() {
  case "$1" in
    full_basic)      printf '%s\n' '2|1|1e-3|1.0,1.0|component|flow|on|position_mean|guided_r8' ;;
    minus_flow)      printf '%s\n' '2|1|1e-3|1.0,1.0|component|identity|on|position_mean|guided_r8' ;;
    minus_loo)       printf '%s\n' '2|1|1e-3|1.0,1.0|component|flow|off|position_mean|guided_r8' ;;
    minus_dvt)       printf '%s\n' '2|1|1e-3|1.0,1.0|component|flow|on|none|guided_r8' ;;
    minus_rgb)       printf '%s\n' '2|1|1e-3|1.0,1.0|component|flow|on|position_mean|none' ;;
    opt_e1_lr2) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|1e-3|1.0,1.0|optimization|flow|off|position_mean|guided_r8|1|2e-4|5,11,17,23" ;;
    opt_e3_lr1) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|1e-3|1.0,1.0|optimization|flow|off|position_mean|guided_r8|3|1e-4|5,11,17,23" ;;
    opt_e3_lr2) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|1e-3|1.0,1.0|optimization|flow|off|position_mean|guided_r8|3|2e-4|5,11,17,23" ;;
    opt_e3_lr5) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|1e-3|1.0,1.0|optimization|flow|off|position_mean|guided_r8|3|5e-4|5,11,17,23" ;;
    opt_e5_lr2) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|1e-3|1.0,1.0|optimization|flow|off|position_mean|guided_r8|5|2e-4|5,11,17,23" ;;
    reg_base) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|1e-3|1.0,1.0|regularization|flow|off|position_mean|guided_r8|${SELECTED_EPOCHS}|${SELECTED_LR}|5,11,17,23" ;;
    reg_ld2) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|2e-2|1.0,1.0|regularization|flow|off|position_mean|guided_r8|${SELECTED_EPOCHS}|${SELECTED_LR}|5,11,17,23" ;;
    reg_b08) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|1e-3|0.8,1.2|regularization|flow|off|position_mean|guided_r8|${SELECTED_EPOCHS}|${SELECTED_LR}|5,11,17,23" ;;
    reg_ld2_b08) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|2e-2|0.8,1.2|regularization|flow|off|position_mean|guided_r8|${SELECTED_EPOCHS}|${SELECTED_LR}|5,11,17,23" ;;
    layer_early) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|${SELECTED_LOGDET}|${SELECTED_BRIGHTNESS}|layers|flow|off|position_mean|guided_r8|${SELECTED_EPOCHS}|${SELECTED_LR}|2,5,8,11" ;;
    layer_current) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|${SELECTED_LOGDET}|${SELECTED_BRIGHTNESS}|layers|flow|off|position_mean|guided_r8|${SELECTED_EPOCHS}|${SELECTED_LR}|5,11,17,23" ;;
    layer_midlate) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|${SELECTED_LOGDET}|${SELECTED_BRIGHTNESS}|layers|flow|off|position_mean|guided_r8|${SELECTED_EPOCHS}|${SELECTED_LR}|8,13,18,23" ;;
    layer_late) printf '%s\n' "${SELECTED_DEPTH}|${SELECTED_WIDTH}|${SELECTED_LOGDET}|${SELECTED_BRIGHTNESS}|layers|flow|off|position_mean|guided_r8|${SELECTED_EPOCHS}|${SELECTED_LR}|11,15,19,23" ;;
    cap_d1_w1)       printf '%s\n' '1|1|1e-3|1.0,1.0|capacity|flow|off|position_mean|guided_r8' ;;
    cap_d1_w2)       printf '%s\n' '1|2|1e-3|1.0,1.0|capacity|flow|off|position_mean|guided_r8' ;;
    cap_d2_w1)       printf '%s\n' '2|1|1e-3|1.0,1.0|capacity|flow|off|position_mean|guided_r8' ;;
    cap_d2_w2)       printf '%s\n' '2|2|1e-3|1.0,1.0|capacity|flow|off|position_mean|guided_r8' ;;
    cap_d4_w1)       printf '%s\n' '4|1|1e-3|1.0,1.0|capacity|flow|off|position_mean|guided_r8' ;;
    cap_d4_w2)       printf '%s\n' '4|2|1e-3|1.0,1.0|capacity|flow|off|position_mean|guided_r8' ;;
    reg_ld1e3_b08)   printf '%s\n' '2|1|1e-3|0.8,1.2|regularization|flow|off|position_mean|guided_r8' ;;
    reg_ld2e2_b10)   printf '%s\n' '2|1|2e-2|1.0,1.0|regularization|flow|off|position_mean|guided_r8' ;;
    reg_ld2e2_b08)   printf '%s\n' '2|1|2e-2|0.8,1.2|regularization|flow|off|position_mean|guided_r8' ;;
    *) echo "unknown config: $1" >&2; return 2 ;;
  esac
}

wait_for_gpu() {
  local gpu="$1" used
  while true; do
    used="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= GPU_IDLE_MEMORY_MIB )); then
      return 0
    fi
    printf 'WAIT gpu=%s memory_used_mib=%s\n' "${gpu}" "${used}"
    sleep "${GPU_IDLE_POLL_SECONDS}"
  done
}

run_object() {
  local gpu="$1" config_id="$2" object="$3" depth="$4" width="$5" logdet="$6" brightness="$7"
  local flow_mode="$8" loo_mode="$9" dvt_mode="${10}" rgb_mode="${11}"
  local epochs="${12:-3}" lr="${13:-2e-4}" layers="${14:-5,11,17,23}"
  local config_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}/configs/${config_id}"
  local shard_root="${config_root}/chunks/${object}"
  local raw_root="${shard_root}/raw"
  local guided_root="${shard_root}/guided_r8_morph"
  local support_json="${REMOTE_RESULTS_ROOT}/${RUN_NAME}/support_paths_${HOST_TAG}.json"
  local log_root="${config_root}/logs"
  mkdir -p "${log_root}"
  if [[ -f "${guided_root}/metrics.json" && -f "${raw_root}/run_manifest.json" ]]; then
    printf 'SKIP_COMPLETE config=%s object=%s\n' "${config_id}" "${object}"
    return 0
  fi
  wait_for_gpu "${gpu}"
  env CUDA_VISIBLE_DEVICES="${gpu}" python3 "${REMOTE_REPO}/scripts/run_flow_tte_mvtec_ad2.py" \
    --data-root "${DATA_ROOT}" --project-root /workspace --fsad-root "${REMOTE_REPO}" \
    --shots 16 --seed 0 --device cuda --flow-epochs "${epochs}" \
    --coupling-layers "${depth}" --hidden-multiplier "${width}" \
    --flow-lr "${lr}" --flow-clamp 1.9 --tail-weight 0.3 --tail-top-k-ratio 0.05 \
    --lambda-logdet "${logdet}" --density-quantile 0.90 --expansion-budget 1.0 \
    --distance-weight 1.0 --density-weight 0.25 --score-mode latent_distance \
    --residual-weight 0.25 --top-percent 0.01 --query-chunk-size 512 \
    --pro-integration-limit 0.05 --rgb-guide none --binary-postprocess closefill_erode \
    --morphology-line-length 17 --morphology-angle-count 16 \
    --backbone-model dinov2_vitl14 --backbone-resolution 0 \
    --feature-layers "${layers}" --tile-patch-size 0 --tile-overlap 0 \
    --image-resize-factor 1.0 --support-brightness-range "${brightness}" \
    --support-selection "fixed_json=${support_json}" --support-selection-seed 0 \
    --support-transforms identity --feature-fusion layer_norm_mean \
    --normality-mode fused --context-source none --flow-context-source auto \
    --memory-context-source auto --context-mode none --context-weight 0.0 \
    --context-top-m 1 --calibration-sample-size 0 --flow-condition-mode none \
    --transformer-context-mode none --flow-transform-mode "${flow_mode}" \
    --loo-standardization "${loo_mode}" \
    --dvt-denoise-mode "${dvt_mode}" --dvt-denoise-alpha 1.0 \
    --score-field-calibration-mode none --score-field-calibration-alpha 1.0 \
    --score-field-position-std-floor 0.25 --score-field-foreground-mode none \
    --score-field-foreground-quantile 0.20 --score-field-background-multiplier 0.50 \
    --score-field-foreground-smooth-kernel 5 --score-field-support-score-quantile 0.90 \
    --output-root "${raw_root}" --objects "${object}" \
    >"${log_root}/${object}_raw.log" 2>&1

  cp "${raw_root}/run_manifest.json" "${raw_root}/upstream_run_manifest.json"
  cp "${raw_root}/metrics.json" "${raw_root}/upstream_metrics.json"
  cp "${raw_root}/metrics_seed=0.json" "${raw_root}/upstream_metrics_seed=0.json"
  env CUDA_VISIBLE_DEVICES="${gpu}" python3 "${REMOTE_REPO}/scripts/run_flow_tte_mvtecad2_guided_refinement.py" \
    --data-root "${DATA_ROOT}" --source-root "${raw_root}" --output-root "${raw_root}" \
    --objects "${object}" --seed 0 --binary-postprocess closefill_erode \
    --morphology-line-length 17 --morphology-angle-count 16 \
    --cleanup-source-maps --cleanup-output-maps \
    >"${log_root}/${object}_guided.log" 2>&1
  mkdir -p "${guided_root}"
  cp "${raw_root}/run_manifest.json" "${guided_root}/run_manifest.json"
  cp "${raw_root}/metrics.json" "${guided_root}/metrics.json"
  cp "${raw_root}/metrics_seed=0.json" "${guided_root}/metrics_seed=0.json"
  cp "${raw_root}/cleanup_evidence.txt" "${guided_root}/cleanup_evidence.txt"
  mv "${raw_root}/upstream_run_manifest.json" "${raw_root}/run_manifest.json"
  mv "${raw_root}/upstream_metrics.json" "${raw_root}/metrics.json"
  mv "${raw_root}/upstream_metrics_seed=0.json" "${raw_root}/metrics_seed=0.json"
}

run_config() {
  local gpu="$1" config_id="$2" values depth width logdet brightness family
  local flow_mode loo_mode dvt_mode rgb_mode epochs lr layers
  local config_root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}/configs/${config_id}"
  if [[ -f "${config_root}/config_complete.txt" ]]; then
    printf 'SKIP_COMPLETE config=%s\n' "${config_id}"
    return 0
  fi
  values="$(config_values "${config_id}")"
  IFS='|' read -r depth width logdet brightness family flow_mode loo_mode dvt_mode rgb_mode epochs lr layers <<<"${values}"
  epochs="${epochs:-3}"; lr="${lr:-2e-4}"; layers="${layers:-5,11,17,23}"
  mkdir -p "${config_root}"
  printf 'config=%s\ndepth=%s\nwidth=%s\nlambda_logdet=%s\nbrightness=%s\nfamily=%s\nflow_mode=%s\nloo=%s\ndvt=%s\nrgb=%s\n' \
    "${config_id}" "${depth}" "${width}" "${logdet}" "${brightness}" "${family}" \
    "${flow_mode}" "${loo_mode}" "${dvt_mode}" "${rgb_mode}" \
    >"${config_root}/config_contract.txt"
  local object
  for object in "${OBJECTS[@]}"; do
    run_object "${gpu}" "${config_id}" "${object}" "${depth}" "${width}" "${logdet}" "${brightness}" \
      "${flow_mode}" "${loo_mode}" "${dvt_mode}" "${rgb_mode}" \
      "${epochs}" "${lr}" "${layers}"
  done
  python3 "${REMOTE_REPO}/scripts/aggregate_flow_tte_ad2_guided_chunks.py" \
    --root "${config_root}" >"${config_root}/aggregate.log" 2>&1
  if find "${config_root}" -type d \( -name anomaly_maps -o -name anomaly_maps_guided_r8 \) -print -quit | grep -q .; then
    echo "dense maps remain after config ${config_id}" >&2
    return 1
  fi
  printf 'config=%s\ncompleted_utc=%s\n' "${config_id}" "$(date -u +%FT%TZ)" \
    >"${config_root}/config_complete.txt"
}

run_worker() {
  local gpu="$1"; shift
  local config_id
  for config_id in "$@"; do
    run_config "${gpu}" "${config_id}"
  done
}

run_config_shard() {
  local gpu="$1" config_id="$2"; shift 2
  local values depth width logdet brightness family flow_mode loo_mode dvt_mode rgb_mode epochs lr layers object
  values="$(config_values "${config_id}")"
  IFS='|' read -r depth width logdet brightness family flow_mode loo_mode dvt_mode rgb_mode epochs lr layers <<<"${values}"
  epochs="${epochs:-3}"; lr="${lr:-2e-4}"; layers="${layers:-5,11,17,23}"
  mkdir -p "${REMOTE_RESULTS_ROOT}/${RUN_NAME}/configs/${config_id}"
  for object in "$@"; do
    run_object "${gpu}" "${config_id}" "${object}" "${depth}" "${width}" "${logdet}" "${brightness}" \
      "${flow_mode}" "${loo_mode}" "${dvt_mode}" "${rgb_mode}" \
      "${epochs}" "${lr}" "${layers}"
  done
}

internal_controller() {
  HOST_TAG="$1"
  export HOST_TAG
  local root="${REMOTE_RESULTS_ROOT}/${RUN_NAME}"
  mkdir -p "${root}/logs"
  echo "$$" >"${root}/${HOST_TAG}_controller.pid"
  python3 "${REMOTE_REPO}/scripts/rebase_fixed_support_json.py" \
    --input "${REMOTE_REPO}/${SUPPORT_JSON_REL}" --data-root "${DATA_ROOT}" \
    --objects "$(IFS=,; echo "${OBJECTS[*]}")" \
    --output "${root}/support_paths_${HOST_TAG}.json"
  local pids=()
  case "${HOST_TAG}" in
    dsba3)
      if [[ "${EXPERIMENT_KIND}" == component ]]; then
        run_worker 0 full_basic >"${root}/logs/dsba3_gpu0.log" 2>&1 & pids+=("$!")
        run_worker 1 minus_flow >"${root}/logs/dsba3_gpu1.log" 2>&1 & pids+=("$!")
        run_worker 2 minus_loo >"${root}/logs/dsba3_gpu2.log" 2>&1 & pids+=("$!")
        run_worker 3 minus_dvt >"${root}/logs/dsba3_gpu3.log" 2>&1 & pids+=("$!")
      elif [[ "${EXPERIMENT_KIND}" == hparam_stage1 ]]; then
        run_worker 0 cap_d1_w1 >"${root}/logs/dsba3_gpu0.log" 2>&1 & pids+=("$!")
        run_worker 1 cap_d1_w2 >"${root}/logs/dsba3_gpu1.log" 2>&1 & pids+=("$!")
        run_worker 2 cap_d2_w1 >"${root}/logs/dsba3_gpu2.log" 2>&1 & pids+=("$!")
        run_worker 3 cap_d2_w2 >"${root}/logs/dsba3_gpu3.log" 2>&1 & pids+=("$!")
      elif [[ "${EXPERIMENT_KIND}" == hparam_stage2 ]]; then
        run_worker 0 opt_e1_lr2 >"${root}/logs/dsba3_gpu0.log" 2>&1 & pids+=("$!")
        run_worker 1 opt_e3_lr1 >"${root}/logs/dsba3_gpu1.log" 2>&1 & pids+=("$!")
        run_worker 2 opt_e3_lr2 >"${root}/logs/dsba3_gpu2.log" 2>&1 & pids+=("$!")
        run_worker 3 opt_e3_lr5 >"${root}/logs/dsba3_gpu3.log" 2>&1 & pids+=("$!")
      elif [[ "${EXPERIMENT_KIND}" == hparam_stage3 ]]; then
        run_worker 0 reg_base >"${root}/logs/dsba3_gpu0.log" 2>&1 & pids+=("$!")
        run_worker 1 reg_ld2 >"${root}/logs/dsba3_gpu1.log" 2>&1 & pids+=("$!")
        run_worker 2 reg_b08 >"${root}/logs/dsba3_gpu2.log" 2>&1 & pids+=("$!")
        run_worker 3 reg_ld2_b08 >"${root}/logs/dsba3_gpu3.log" 2>&1 & pids+=("$!")
      elif [[ "${EXPERIMENT_KIND}" == hparam_stage4 ]]; then
        run_worker 0 layer_early >"${root}/logs/dsba3_gpu0.log" 2>&1 & pids+=("$!")
        run_worker 1 layer_current >"${root}/logs/dsba3_gpu1.log" 2>&1 & pids+=("$!")
        run_worker 2 layer_midlate >"${root}/logs/dsba3_gpu2.log" 2>&1 & pids+=("$!")
        run_worker 3 layer_late >"${root}/logs/dsba3_gpu3.log" 2>&1 & pids+=("$!")
      elif [[ "${EXPERIMENT_KIND}" == hparam_stage2_e5 ]]; then
        echo "NO_ASSIGNED_CONFIGS host=dsba3 stage=${EXPERIMENT_KIND}"
      else
        run_worker 0 cap_d1_w1 reg_ld2e2_b08 >"${root}/logs/dsba3_gpu0.log" 2>&1 & pids+=("$!")
        run_worker 1 cap_d1_w2 reg_ld2e2_b10 >"${root}/logs/dsba3_gpu1.log" 2>&1 & pids+=("$!")
        run_worker 2 cap_d2_w1 reg_ld1e3_b08 >"${root}/logs/dsba3_gpu2.log" 2>&1 & pids+=("$!")
        run_worker 3 cap_d2_w2 >"${root}/logs/dsba3_gpu3.log" 2>&1 & pids+=("$!")
      fi
      ;;
    dsba5)
      if [[ "${EXPERIMENT_KIND}" == component ]]; then
        run_config_shard 0 minus_rgb can fabric fruit_jelly rice >"${root}/logs/dsba5_gpu0.log" 2>&1 & pids+=("$!")
        run_config_shard 1 minus_rgb vial wallplugs walnuts sheet_metal >"${root}/logs/dsba5_gpu1.log" 2>&1 & pids+=("$!")
      elif [[ "${EXPERIMENT_KIND}" == hparam_stage1 ]]; then
        run_worker 0 cap_d4_w1 >"${root}/logs/dsba5_gpu0.log" 2>&1 & pids+=("$!")
        run_worker 1 cap_d4_w2 >"${root}/logs/dsba5_gpu1.log" 2>&1 & pids+=("$!")
      elif [[ "${EXPERIMENT_KIND}" == hparam_stage2 ]]; then
        run_config_shard 0 opt_e5_lr2 can fabric fruit_jelly rice >"${root}/logs/dsba5_gpu0.log" 2>&1 & pids+=("$!")
        run_config_shard 1 opt_e5_lr2 vial wallplugs walnuts sheet_metal >"${root}/logs/dsba5_gpu1.log" 2>&1 & pids+=("$!")
      elif [[ "${EXPERIMENT_KIND}" == hparam_stage2_e5 ]]; then
        run_config_shard 0 opt_e5_lr2 can fabric fruit_jelly rice >"${root}/logs/dsba5_gpu0.log" 2>&1 & pids+=("$!")
        run_config_shard 1 opt_e5_lr2 vial wallplugs walnuts sheet_metal >"${root}/logs/dsba5_gpu1.log" 2>&1 & pids+=("$!")
      elif [[ "${EXPERIMENT_KIND}" == hparam_stage3 || "${EXPERIMENT_KIND}" == hparam_stage4 ]]; then
        echo "NO_ASSIGNED_CONFIGS host=dsba5 stage=${EXPERIMENT_KIND}"
      else
        run_worker 0 cap_d4_w1 >"${root}/logs/dsba5_gpu0.log" 2>&1 & pids+=("$!")
        run_worker 1 cap_d4_w2 >"${root}/logs/dsba5_gpu1.log" 2>&1 & pids+=("$!")
      fi
      ;;
    *) echo "unknown host tag: ${HOST_TAG}" >&2; return 2 ;;
  esac
  local status=0 pid
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  [[ "${status}" -eq 0 ]] || return "${status}"
  if [[ "${HOST_TAG}" == dsba5 && "${EXPERIMENT_KIND}" == component ]]; then
    local config_root="${root}/configs/minus_rgb"
    python3 "${REMOTE_REPO}/scripts/aggregate_flow_tte_ad2_guided_chunks.py" \
      --root "${config_root}" >"${config_root}/aggregate.log" 2>&1
    printf 'config=minus_rgb\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" \
      >"${config_root}/config_complete.txt"
  fi
  if [[ "${HOST_TAG}" == dsba5 && ( "${EXPERIMENT_KIND}" == hparam_stage2 || "${EXPERIMENT_KIND}" == hparam_stage2_e5 ) ]]; then
    local config_root="${root}/configs/opt_e5_lr2"
    python3 "${REMOTE_REPO}/scripts/aggregate_flow_tte_ad2_guided_chunks.py" \
      --root "${config_root}" >"${config_root}/aggregate.log" 2>&1
    printf 'config=opt_e5_lr2\ncompleted_utc=%s\n' "$(date -u +%FT%TZ)" \
      >"${config_root}/config_complete.txt"
  fi
  printf 'host=%s\ncompleted_utc=%s\n' "${HOST_TAG}" "$(date -u +%FT%TZ)" \
    >"${root}/${HOST_TAG}_complete.txt"
}

status_host() {
  local tag="$1"
  local preset="${ROOT_DIR}/configs/remote/${tag}.env"
  load_preset "${preset}"
  ssh_remote "${REMOTE_HOST}" "docker exec '${CONTAINER_NAME}' bash -lc '
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    echo HOST=${tag}
    test -f \"\${root}/${tag}_complete.txt\" && echo COMPLETE=yes || echo COMPLETE=no
    echo CONFIGS_COMPLETE=\$(find \"\${root}/configs\" -name config_complete.txt 2>/dev/null | wc -l)
    echo OBJECTS_GUIDED=\$(find \"\${root}/configs\" -path \"*/guided_r8_morph/metrics.json\" 2>/dev/null | wc -l)
    echo ACTIVE=\$(ps -eo cmd | grep run_flow_tte_mvtec_ad2.py | grep -v grep | wc -l)
    if test -s \"\${root}/${tag}_controller.pid\"; then
      pid=\$(cat \"\${root}/${tag}_controller.pid\")
      echo CONTROLLER=\$(ps -o stat= -p \"\${pid}\")
      echo CONTROLLER_ELAPSED=\$(ps -o etime= -p \"\${pid}\" | xargs)
      echo CHILDREN=\$(ps --ppid \"\${pid}\" -o pid= | wc -l)
    fi
    echo ERRORS=\$(grep -R -l -E \"Traceback|CUDA out of memory|RuntimeError:\" \
      \"\${root}/logs\" \"\${root}/configs\" 2>/dev/null | wc -l)
  '"
}

pause_host() {
  local tag="$1"
  local preset="${ROOT_DIR}/configs/remote/${tag}.env"
  load_preset "${preset}"
  ssh_remote "${REMOTE_HOST}" "docker exec '${CONTAINER_NAME}' bash -lc '
    root=${REMOTE_RESULTS_ROOT}/${RUN_NAME}
    test -d \"\${root}\" || exit 0
    ps -eo pid=,comm=,args= | awk -v root=\"\${root}\" \
      '\''\$2 == \"python3\" && index(\$0, root) && index(\$0, \"run_flow_tte_mvtec_ad2.py\") {print \$1}'\'' \
      | xargs -r kill -TERM 2>/dev/null || true
    if test -s \"\${root}/${tag}_controller.pid\"; then
      pid=\$(cat \"\${root}/${tag}_controller.pid\")
      pkill -TERM -P \"\${pid}\" 2>/dev/null || true
      kill -TERM \"\${pid}\" 2>/dev/null || true
    fi
    date -u +%FT%TZ >\"\${root}/${tag}_paused_for_component_ablation.txt\"
    echo PAUSED host=${tag}
  '"
}

pull_host() {
  local tag="$1"
  local preset="${ROOT_DIR}/configs/remote/${tag}.env"
  local local_root="${ROOT_DIR}/results/remote_runs/${tag}/${RUN_NAME}"
  load_preset "${preset}"
  mkdir -p "${local_root}"
  ssh_remote "${REMOTE_HOST}" \
    "docker exec '${CONTAINER_NAME}' tar -C '${REMOTE_RESULTS_ROOT}/${RUN_NAME}' -cf - ." \
    | tar -C "${local_root}" -xf -
  printf 'PULLED host=%s root=%s\n' "${tag}" "${local_root}"
}

case "${MODE}" in
  plan) print_plan ;;
  start)
    print_plan
    start_host dsba3
    start_host dsba5
    ;;
  status)
    status_host dsba3
    status_host dsba5
    ;;
  pause)
    pause_host dsba3
    pause_host dsba5
    ;;
  restart-dsba5)
    pause_host dsba5
    start_host dsba5
    ;;
  pull)
    pull_host dsba3
    pull_host dsba5
    ;;
  internal-controller)
    [[ "$#" -eq 2 ]] || { echo "internal-controller requires host tag" >&2; exit 2; }
    internal_controller "$2"
    ;;
  *) echo "usage: $0 {plan|start|status|pause|restart-dsba5|pull}" >&2; exit 2 ;;
esac
