#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-sdig_nodyr_stage_a_all8_20260717_v1}"
MODE="${MODE:-full}"
FSAD_ROOT="${FSAD_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_ROOT}/${RUN_NAME}}"
SUPPORT_JSON="${SUPPORT_JSON:-${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flow_latentbank_no_tte_fixed_reference_dinov3/superad16_dinov2_reference_paths.json}"

if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "MODE must be smoke or full" >&2
  exit 2
fi
if [[ "$(python3 -c 'import torch; print(torch.cuda.device_count())')" -lt 4 ]]; then
  echo "four visible CUDA devices are required" >&2
  exit 3
fi

objects=(can fabric fruit_jelly rice sheet_metal vial wallplugs walnuts)
for object_name in "${objects[@]}"; do
  test -n "$(find "${DATA_ROOT}/${object_name}/train/good" -maxdepth 1 -type f -name '*.png' -print -quit)"
  test -n "$(find "${DATA_ROOT}/${object_name}/test_public/good" -maxdepth 1 -type f -name '*.png' -print -quit)"
  test -n "$(find "${DATA_ROOT}/${object_name}/test_public/bad" -maxdepth 1 -type f -name '*.png' -print -quit)"
done
test -f "${SUPPORT_JSON}"
mkdir -p "${OUTPUT_ROOT}/chunks"

{
  printf 'run_name=%s\nmode=%s\n' "${RUN_NAME}" "${MODE}"
  printf 'target_dataset=MVTec AD2 single-image\n'
  printf 'data_root=%s\n' "${DATA_ROOT}"
  printf 'host_gpu_ids=0,1,2,3\ncontainer_cuda_slots=0_after_per_process_mask\n'
  printf 'support_json=%s\n' "${SUPPORT_JSON}"
  sha256sum "${SUPPORT_JSON}" "${FSAD_ROOT}/scripts/run_sdig_nodyr_ad2.py" \
    "${FSAD_ROOT}/src/flow_tte/nodyr.py"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
  for object_name in "${objects[@]}"; do
    printf 'population_%s_train_good=%s\n' "${object_name}" \
      "$(find "${DATA_ROOT}/${object_name}/train/good" -maxdepth 1 -type f -name '*.png' | wc -l)"
    printf 'population_%s_test_good=%s\n' "${object_name}" \
      "$(find "${DATA_ROOT}/${object_name}/test_public/good" -maxdepth 1 -type f -name '*.png' | wc -l)"
    printf 'population_%s_test_bad=%s\n' "${object_name}" \
      "$(find "${DATA_ROOT}/${object_name}/test_public/bad" -maxdepth 1 -type f -name '*.png' | wc -l)"
  done
} >"${OUTPUT_ROOT}/preflight.txt"

run_one() {
  local gpu="$1"
  local object_name="$2"
  local limit_args=()
  local cleanup_args=(--cleanup-maps)
  if [[ "${MODE}" == "smoke" ]]; then
    limit_args=(--source-ids 000)
  fi
  local chunk_root="${OUTPUT_ROOT}/chunks/gpu${gpu}_${object_name}"
  mkdir -p "${chunk_root}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 "${FSAD_ROOT}/scripts/run_sdig_nodyr_ad2.py" \
    --data-root "${DATA_ROOT}" \
    --output-root "${chunk_root}" \
    --support-json "${SUPPORT_JSON}" \
    --objects "${object_name}" \
    --device cuda \
    --backbone-model dinov2_vitl14 \
    --layers 15,23 \
    --ridge 0.01 \
    --seed 0 \
    --host-gpu-id "${gpu}" \
    --container-cuda-slot 0 \
    "${limit_args[@]}" \
    "${cleanup_args[@]}" \
    >"${chunk_root}/run.log" 2>&1
}

run_group() {
  local gpu="$1"
  shift
  local object_name
  for object_name in "$@"; do
    run_one "${gpu}" "${object_name}"
  done
}

if [[ "${MODE}" == "smoke" ]]; then
  run_group 0 can & p0=$!
  run_group 1 fabric & p1=$!
  run_group 2 rice & p2=$!
  run_group 3 vial & p3=$!
else
  run_group 0 can fruit_jelly & p0=$!
  run_group 1 fabric sheet_metal & p1=$!
  run_group 2 rice wallplugs & p2=$!
  run_group 3 vial walnuts & p3=$!
fi

status=0
for pid in "${p0}" "${p1}" "${p2}" "${p3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more NODyR shards failed" >&2
  exit "${status}"
fi

aggregate_args=()
if [[ "${MODE}" == "full" ]]; then
  aggregate_args=(--require-all8)
fi
python3 "${FSAD_ROOT}/scripts/aggregate_sdig_nodyr_stage_a.py" \
  --run-root "${OUTPUT_ROOT}" "${aggregate_args[@]}" \
  >"${OUTPUT_ROOT}/aggregate.log" 2>&1
find "${OUTPUT_ROOT}" -type d -name anomaly_maps -prune -print >"${OUTPUT_ROOT}/remaining_anomaly_maps.txt"
test ! -s "${OUTPUT_ROOT}/remaining_anomaly_maps.txt"
find "${OUTPUT_ROOT}" -type d -name patch_score_cache -prune -print >"${OUTPUT_ROOT}/remaining_patch_score_cache.txt"
test ! -s "${OUTPUT_ROOT}/remaining_patch_score_cache.txt"
printf 'run_name=%s\nmode=%s\nstatus=complete\ncleanup_anomaly_maps=true\n' \
  "${RUN_NAME}" "${MODE}" >"${OUTPUT_ROOT}/complete.txt"
