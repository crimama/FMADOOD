#!/usr/bin/env bash
set -euo pipefail

# Generic multi-device launcher. Each lane owns disjoint objects or stages
# and runs its configurations through scripts/experiments.py.
# It intentionally does not know method names, queue policy, or ledger format.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="local-docker"
CONTAINER_NAME="${CONTAINER_NAME:-fsad-tta-local-bcma}"
DATA_ROOT="/data/mvtec_ad_2"
OUTPUT_ROOT=""
LOG_DIR=""
SPLIT="test_public_seen_development"
TIMEOUT=3600
WORKERS=1
SHARD_BY=objects
REMOTE_SHARD_BY=objects
RESUME=0
CONFIGS=()
LABELS=()
OBJECTS=()
DEVICES=()
LOCAL_DEVICE=""
LOCAL_OBJECTS=()
REMOTE_DEVICES=()
REMOTE_OBJECTS=()
REMOTE_DATA_ROOT=""
AFTER_STATUS=()
ENTRYPOINT="run.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_experiments.sh [options]

Required:
  --configs C...       configuration paths, terminated by the next option
  --labels L...        one label per configuration
  --objects O...       object names, partitioned across devices
  --devices D...       one local device or remote UUID per lane
  --output-root PATH   output root visible in the execution environment

Options:
  --backend local-docker|remote   execution backend (default: local-docker)
  --backend both                  launch local and remote lanes from one command
  --container NAME                existing local/remote container name
  --data-root PATH                dataset path in the container
  --remote-data-root PATH         dataset path in the remote container (both mode)
  --log-dir PATH                  host directory for launcher logs
  --split NAME                    declared split (default: test_public_seen_development)
  --timeout SECONDS               per-object timeout (default: 3600)
  --workers N                     workers inside each lane (default: 1)
  --shard-by objects|stages        partition objects or configs across devices
  --remote-shard-by objects|stages remote partition in both mode (default: objects)
  --resume                        resume an existing invalid lane
  --after-status PATH...          wait for successful predecessor status files
  --entrypoint PATH               CLI entrypoint (default: run.py)

For --backend both, also provide --local-device, --local-objects,
--remote-devices, and --remote-objects. The caller must source the remote
preset first; the remote child uses the existing run_remote.sh helper.

Remote mode expects the designated configs/remote/*.env preset to be sourced and
uses scripts/run_remote.sh once, with all requested UUIDs exposed to that call.
No container is created, removed, or rebuilt by this launcher.
EOF
}

die() { echo "run_experiments.sh: $*" >&2; exit 2; }

check_unique() {
  local label="$1"; shift
  local -A seen=()
  local value
  for value in "$@"; do
    [[ -n "$value" ]] || die "empty $label"
    [[ -z "${seen[$value]+present}" ]] || die "duplicate $label: $value"
    seen[$value]=1
  done
}

take_list() {
  local name="$1"; shift
  local -n target="$name"
  target=()
  while [[ $# -gt 0 && "$1" != --* ]]; do target+=("$1"); shift; done
  [[ ${#target[@]} -gt 0 ]] || die "--${name,,} requires at least one value"
  PARSE_REST=("$@")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --configs|--labels|--objects|--devices|--local-objects|--remote-devices|--remote-objects|--after-status)
      key="${1#--}"; shift
      case "$key" in
        configs) target=CONFIGS ;;
        labels) target=LABELS ;;
        objects) target=OBJECTS ;;
        devices) target=DEVICES ;;
        local-objects) target=LOCAL_OBJECTS ;;
        remote-devices) target=REMOTE_DEVICES ;;
        remote-objects) target=REMOTE_OBJECTS ;;
        after-status) target=AFTER_STATUS ;;
      esac
      take_list "$target" "$@"; set -- "${PARSE_REST[@]}"; unset key target; ;;
    --backend) [[ $# -ge 2 ]] || die "--backend needs a value"; BACKEND="$2"; shift 2 ;;
    --entrypoint) [[ $# -ge 2 ]] || die "--entrypoint needs a value"; ENTRYPOINT="$2"; shift 2 ;;
    --container) [[ $# -ge 2 ]] || die "--container needs a value"; CONTAINER_NAME="$2"; shift 2 ;;
    --local-device) [[ $# -ge 2 ]] || die "--local-device needs a value"; LOCAL_DEVICE="$2"; shift 2 ;;
    --data-root) [[ $# -ge 2 ]] || die "--data-root needs a value"; DATA_ROOT="$2"; shift 2 ;;
    --remote-data-root) [[ $# -ge 2 ]] || die "--remote-data-root needs a value"; REMOTE_DATA_ROOT="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || die "--output-root needs a value"; OUTPUT_ROOT="$2"; shift 2 ;;
    --log-dir) [[ $# -ge 2 ]] || die "--log-dir needs a value"; LOG_DIR="$2"; shift 2 ;;
    --split) [[ $# -ge 2 ]] || die "--split needs a value"; SPLIT="$2"; shift 2 ;;
    --timeout) [[ $# -ge 2 ]] || die "--timeout needs a value"; TIMEOUT="$2"; shift 2 ;;
    --workers) [[ $# -ge 2 ]] || die "--workers needs a value"; WORKERS="$2"; shift 2 ;;
    --shard-by) [[ $# -ge 2 ]] || die "--shard-by needs a value"; SHARD_BY="$2"; shift 2 ;;
    --remote-shard-by) [[ $# -ge 2 ]] || die "--remote-shard-by needs a value"; REMOTE_SHARD_BY="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

[[ ${#CONFIGS[@]} -gt 0 ]] || die "--configs is required"
[[ ${#LABELS[@]} -eq ${#CONFIGS[@]} ]] || die "--labels count must equal --configs count"
[[ -n "$OUTPUT_ROOT" ]] || die "--output-root is required"
[[ -n "$LOG_DIR" ]] || LOG_DIR="$ROOT_DIR/results/experiment_launch"
[[ "$BACKEND" == local-docker || "$BACKEND" == remote || "$BACKEND" == both ]] || die "unsupported backend: $BACKEND"
[[ "$TIMEOUT" =~ ^[0-9]+$ && "$TIMEOUT" -gt 0 ]] || die "timeout must be a positive integer"
[[ "$WORKERS" =~ ^[0-9]+$ && "$WORKERS" -gt 0 ]] || die "workers must be a positive integer"
[[ "$SHARD_BY" == objects || "$SHARD_BY" == stages ]] || die "invalid --shard-by"
[[ "$REMOTE_SHARD_BY" == objects || "$REMOTE_SHARD_BY" == stages ]] || die "invalid --remote-shard-by"
cd "$ROOT_DIR"
check_unique label "${LABELS[@]}"
for config in "${CONFIGS[@]}"; do [[ -f "$config" ]] || die "config not found: $config"; done

if [[ "$BACKEND" == both ]]; then
  [[ -n "$LOCAL_DEVICE" ]] || die "--local-device is required for --backend both"
  [[ ${#LOCAL_OBJECTS[@]} -gt 0 ]] || die "--local-objects is required for --backend both"
  [[ ${#REMOTE_DEVICES[@]} -gt 0 ]] || die "--remote-devices is required for --backend both"
  [[ ${#REMOTE_OBJECTS[@]} -gt 0 ]] || die "--remote-objects is required for --backend both"
  REMOTE_DATA_ROOT="${REMOTE_DATA_ROOT:-/data/mvtec_ad_2}"
  check_unique object "${LOCAL_OBJECTS[@]}" "${REMOTE_OBJECTS[@]}"
  check_unique remote-device "${REMOTE_DEVICES[@]}"
  if [[ "$REMOTE_SHARD_BY" == stages ]]; then
    (( ${#REMOTE_DEVICES[@]} <= ${#CONFIGS[@]} )) || die "more remote devices than stages"
  else
    (( ${#REMOTE_DEVICES[@]} <= ${#REMOTE_OBJECTS[@]} )) || die "more remote devices than objects"
  fi
  mkdir -p "$LOG_DIR"
  local_args=(--backend local-docker --container fsad-tta-local-bcma
              --configs "${CONFIGS[@]}" --labels "${LABELS[@]}"
              --objects "${LOCAL_OBJECTS[@]}" --devices "$LOCAL_DEVICE"
              --data-root "$DATA_ROOT" --output-root "$OUTPUT_ROOT/local"
              --log-dir "$LOG_DIR/local" --split "$SPLIT" --timeout "$TIMEOUT" --workers "$WORKERS"
              --shard-by "$SHARD_BY")
  remote_args=(--backend remote --container "$CONTAINER_NAME"
               --configs "${CONFIGS[@]}" --labels "${LABELS[@]}"
               --objects "${REMOTE_OBJECTS[@]}" --devices "${REMOTE_DEVICES[@]}"
               --data-root "$REMOTE_DATA_ROOT" --output-root "$OUTPUT_ROOT/remote"
               --log-dir "$LOG_DIR/remote" --split "$SPLIT" --timeout "$TIMEOUT" --workers "$WORKERS"
               --shard-by "$REMOTE_SHARD_BY")
  (( RESUME == 1 )) && { local_args+=(--resume); remote_args+=(--resume); }
  local_args+=(--entrypoint "$ENTRYPOINT")
  remote_args+=(--entrypoint "$ENTRYPOINT")
  if (( ${#AFTER_STATUS[@]} > 0 )); then
    local_args+=(--after-status "${AFTER_STATUS[@]}")
    remote_args+=(--after-status "${AFTER_STATUS[@]}")
  fi
  echo "Launching local and remote lanes from one command."
  bash "$ROOT_DIR/scripts/run_experiments.sh" "${local_args[@]}" >>"$LOG_DIR/local.launch.log" 2>&1 &
  local_pid=$!
  bash "$ROOT_DIR/scripts/run_experiments.sh" "${remote_args[@]}" >>"$LOG_DIR/remote.launch.log" 2>&1 &
  remote_pid=$!
  set +e
  wait "$local_pid"; local_rc=$?
  wait "$remote_pid"; remote_rc=$?
  set -e
  (( local_rc == 0 && remote_rc == 0 )) || exit 1
  exit 0
fi

[[ ${#OBJECTS[@]} -gt 0 ]] || die "--objects is required"
[[ ${#DEVICES[@]} -gt 0 ]] || die "--devices is required"

if [[ "$BACKEND" == local-docker ]]; then
  command -v docker >/dev/null || die "docker is required for local-docker backend"
  docker inspect "$CONTAINER_NAME" >/dev/null 2>&1 || die "existing container not found: $CONTAINER_NAME"
fi

if [[ "$SHARD_BY" == stages ]]; then
  (( ${#DEVICES[@]} <= ${#CONFIGS[@]} )) || die "more devices than stages"
else
  (( ${#DEVICES[@]} <= ${#OBJECTS[@]} )) || die "more devices than objects"
fi
check_unique object "${OBJECTS[@]}"
check_unique device "${DEVICES[@]}"

config_in_runtime() {
  local path="$1"
  if [[ "$path" == "$ROOT_DIR/"* ]]; then
    # Both the local mount and the remote local-snapshot use the repository
    # root as the working directory. Relative paths therefore survive either
    # runtime without assuming a legacy /workspace checkout.
    printf '%s\n' "${path#"$ROOT_DIR"/}"
  else
    printf '%s\n' "$path"
  fi
}

quote_array() {
  local value
  for value in "$@"; do printf '%q ' "$value"; done
}

LANE_CMDS=()
LANE_LOGS=()
for ((lane=0; lane<${#DEVICES[@]}; lane++)); do
  lane_objects=()
  lane_configs=()
  lane_labels=()
  if [[ "$SHARD_BY" == stages ]]; then
    lane_objects=("${OBJECTS[@]}")
    for ((i=lane; i<${#CONFIGS[@]}; i+=${#DEVICES[@]})); do
      lane_configs+=("${CONFIGS[i]}")
      lane_labels+=("${LABELS[i]}")
    done
  else
    lane_configs=("${CONFIGS[@]}")
    lane_labels=("${LABELS[@]}")
    for ((i=lane; i<${#OBJECTS[@]}; i+=${#DEVICES[@]})); do lane_objects+=("${OBJECTS[i]}"); done
  fi
  (( ${#lane_objects[@]} > 0 )) || continue

  cmd="set -euo pipefail; mkdir -p $(printf '%q' "$OUTPUT_ROOT"); python3 scripts/experiments.py run"
  cmd+=" --configs $(for c in "${lane_configs[@]}"; do printf '%q ' "$(config_in_runtime "$c")"; done)"
  cmd+=" --labels $(quote_array "${lane_labels[@]}")"
  cmd+=" --objects $(quote_array "${lane_objects[@]}")"
  cmd+=" --data-root $(printf '%q' "$DATA_ROOT") --output-dir $(printf '%q' "$OUTPUT_ROOT/lane_$lane")"
  cmd+=" --device cuda:0 --split $(printf '%q' "$SPLIT") --timeout $(printf '%q' "$TIMEOUT") --workers $(printf '%q' "$WORKERS")"
  (( RESUME == 1 )) && cmd+=" --resume"
  cmd+=" --entrypoint $(printf '%q' "$ENTRYPOINT")"
  (( ${#AFTER_STATUS[@]} == 0 )) || cmd+=" --after-status $(quote_array "${AFTER_STATUS[@]}")"
  LANE_CMDS+=("$cmd")
  LANE_LOGS+=("$LOG_DIR/lane_$lane.launch.log")
done

if [[ "$BACKEND" == local-docker ]]; then
  mkdir -p "$LOG_DIR"
  pids=()
  for ((lane=0; lane<${#LANE_CMDS[@]}; lane++)); do
    echo "START lane=$lane device=${DEVICES[lane]}"
    docker exec -e "CUDA_VISIBLE_DEVICES=${DEVICES[lane]}" "$CONTAINER_NAME" \
      bash -c "${LANE_CMDS[lane]}" >>"${LANE_LOGS[lane]}" 2>&1 &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  exit "$rc"
fi

[[ -n "${REMOTE_GPU_DEVICES:-}" ]] || die "REMOTE_GPU_DEVICES must be set in remote mode"
IFS=, read -r -a exposed <<< "$REMOTE_GPU_DEVICES"
[[ ${#exposed[@]} -eq ${#DEVICES[@]} ]] || die "--devices must match REMOTE_GPU_DEVICES count"
check_unique object "${OBJECTS[@]}"
check_unique device "${DEVICES[@]}"
for device in "${DEVICES[@]}"; do
  found=0
  for visible in "${exposed[@]}"; do [[ "$device" == "$visible" ]] && found=1; done
  (( found == 1 )) || die "device is not exposed by REMOTE_GPU_DEVICES: $device"
done
remote_script=$'set -euo pipefail\n'
remote_script+="mkdir -p $(printf '%q' "$OUTPUT_ROOT")"$'\n'
remote_script+=$'pids=()\n'
for ((lane=0; lane<${#LANE_CMDS[@]}; lane++)); do
  remote_log="$OUTPUT_ROOT/lane_$lane.launch.log"
  remote_script+="mkdir -p $(printf '%q' "$(dirname "$remote_log")")"$'\n'
  remote_script+="CUDA_VISIBLE_DEVICES=$(printf '%q' "${DEVICES[lane]}") bash -c $(printf '%q' "${LANE_CMDS[lane]}") >>$(printf '%q' "$remote_log") 2>&1 & pids+=(\$!)"$'\n'
done
remote_script+=$'rc=0\nfor pid in "${pids[@]}"; do wait "$pid" || rc=1; done\nexit "$rc"\n'
export REMOTE_GPU_DEVICES
scripts/run_remote.sh exec -- bash -c "$remote_script"
