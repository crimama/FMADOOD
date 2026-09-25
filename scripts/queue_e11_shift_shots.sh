#!/usr/bin/env bash
set -euo pipefail

# E11 starts only after the four existing E9 VisA chains have finished. Keep
# this scheduler alive (for example under nohup); it never provisions a GPU or
# container, modifies an active checkout, or retries a failed experiment cell.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source configs/remote/dsba3.env
[[ "$REMOTE_CODE_MODE" == local ]] || { echo 'E11 requires local snapshot mode' >&2; exit 2; }

LOCK="$ROOT/results/local_runs/v7_evidence/E11_shift_shots/scheduler.lock"
LOG_DIR="${LOCK%/*}"
mkdir -p "$LOG_DIR"
exec 8>"$LOCK"
flock -n 8 || { echo 'E11 scheduler already active' >&2; exit 1; }

GPUS=(GPU-a164a77e-c3b3-5c6d-d70c-63ee65b2601f
      GPU-bbb3a0e2-cc7d-ba3c-f956-89214c06a1ae
      GPU-eb6182a8-eb72-e8b2-f279-c587d6fa9639
      GPU-7534790f-7c8c-3f2b-5c33-c1b15998ef62)
REMOTE_ROOT=/workspace/results_local_compatible/v7_evidence
LOCAL_ROOT="$ROOT/results/remote_runs/dsba3/v7_evidence"
OUTPUT="$ROOT/results/v7_evidence/E11_shift_shots"
SPLIT=mvtec_ad2_test_public_strata_comparison_no_label_selection
CONFIGS=()
for method in rsg3 superad_dinov3 superad patchcore; do
    for shot in 01 02 04 08; do
        config="configs/benchmarks/v7_evidence/E11_shift_shots/e11_${method}_s${shot}.yaml"
        [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 2; }
        CONFIGS+=("$config")
    done
done

# The probe is read-only. A missing predecessor waits; a failed predecessor or
# an incomplete E11 shard stops the queue instead of silently rerunning cells.
remote_state() {
    local path="$1" mode="${2:-}"
    if [[ "$mode" == E11 ]]; then shift 2; else shift; fi
    REMOTE_GPU_DEVICES="${GPUS[0]}" scripts/run_remote.sh exec -- \
        python3 -c 'import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "status.json"
if not p.is_file():
    print("absent")
else:
    s = json.loads(p.read_text())
    n = len(s.get("completed", []))
    plan = s.get("plan", {})
    total = len(plan.get("labels", [])) * len(plan.get("objects", []))
    state = s.get("state", "unknown")
    if len(sys.argv) > 2 and sys.argv[2] == "E11":
        expected = ["e11_{}_s{}".format(method, shot)
                    for method in ("rsg3", "superad_dinov3", "superad", "patchcore")
                    for shot in ("01", "02", "04", "08")]
        if (plan.get("labels") != expected or plan.get("objects") != sys.argv[3:]
                or s.get("split") != "mvtec_ad2_test_public_strata_comparison_no_label_selection"
                or [pathlib.Path(c).stem for c in plan.get("configs", [])] != expected):
            state = "mismatch"
    print("{} {} {}".format(state, n, total))' "$path" "$mode" "$@" \
        | tail -n 1
}

while :; do
    remaining=0
    for lane in 0 1 2 3; do
        state="$(remote_state "$REMOTE_ROOT/E9_visa/lane$lane")"
        read -r phase done total <<<"$state"
        case "$phase" in
            complete)
                [[ "$total" -gt 0 && "$done" -eq "$total" ]] || {
                    echo "E9_visa lane$lane has incomplete status: $state" >&2; exit 1;
                }
                ;;
            absent|running)
                echo "Waiting for E9_visa lane$lane: $state"
                remaining=$((remaining + 1)) ;;
            *) echo "E9_visa lane$lane failed or has unknown state: $state" >&2; exit 1 ;;
        esac
    done
    (( remaining == 0 )) && break
    sleep 120
done
echo 'All E9_visa lanes complete'

OBJECTS_0=(sheet_metal)
OBJECTS_1=(can fruit_jelly)
OBJECTS_2=(fabric rice)
OBJECTS_3=(vial wallplugs walnuts)
pending=()
for lane in 0 1 2 3; do
    objects_var="OBJECTS_$lane[@]"
    objects=("${!objects_var}")
    state="$(remote_state "$REMOTE_ROOT/E11_shift_shots/lane$lane" E11 "${objects[@]}")"
    read -r phase done total <<<"$state"
    case "$phase" in
        absent) pending+=("$lane") ;;
        complete)
            expected=$(( ${#CONFIGS[@]} * ${#objects[@]} ))
            [[ "$done" -eq "$expected" && "$total" -eq "$expected" ]] || {
                echo "E11 lane$lane has unexpected coverage: $state; expected $expected" >&2; exit 1;
            }
            echo "E11 lane$lane already complete ($state)" ;;
        *) echo "E11 lane$lane has existing partial/failed status: $state" >&2; exit 1 ;;
    esac
done

if (( ${#pending[@]} )); then
    # The helper acquires the exclusive code activation lock. If an unrelated
    # run still holds it, retry without touching that run's checkout.
    while ! update_log="$(scripts/run_remote.sh update-code 2>&1)"; do
        if [[ "$update_log" != *'Experiment/update in progress; code unchanged.'* ]]; then
            echo "$update_log" >&2
            exit 1
        fi
        echo 'Code activation deferred by an active experiment; retrying in 120 seconds' >&2
        sleep 120
    done
    echo "$update_log"
    pids=()
    for lane in "${pending[@]}"; do
        objects_var="OBJECTS_$lane[@]"
        objects=("${!objects_var}")
        (
            export REMOTE_GPU_DEVICES="${GPUS[$lane]}"
            scripts/run_remote.sh exec -- python3 scripts/experiments.py run \
                --configs "${CONFIGS[@]}" \
                --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
                --output-dir "$REMOTE_ROOT/E11_shift_shots/lane$lane" \
                --objects "${objects[@]}" --split "$SPLIT" \
                --device cuda:0 --timeout 10800
        ) >"$LOG_DIR/lane$lane.log" 2>&1 &
        pids+=("$!")
        echo "Started E11 lane$lane on ${GPUS[$lane]} (${objects[*]})"
    done
    failed=0
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    (( failed == 0 )) || { echo "E11 lane failed; inspect $LOG_DIR/lane*.log" >&2; exit 1; }
fi

for lane in 0 1 2 3; do
    objects_var="OBJECTS_$lane[@]"
    objects=("${!objects_var}")
    state="$(remote_state "$REMOTE_ROOT/E11_shift_shots/lane$lane" E11 "${objects[@]}")"
    read -r phase done total <<<"$state"
    expected=$(( ${#CONFIGS[@]} * ${#objects[@]} ))
    [[ "$phase" == complete && "$done" -eq "$expected" && "$total" -eq "$expected" ]] || {
        echo "E11 lane$lane incomplete after run: $state" >&2; exit 1;
    }
done

# Pull once, then one collector writes the canonical ledger for all shards.
scripts/run_remote.sh pull
.venv/bin/python scripts/experiments.py collect --input-dirs \
    "$LOCAL_ROOT/E11_shift_shots/lane0" "$LOCAL_ROOT/E11_shift_shots/lane1" \
    "$LOCAL_ROOT/E11_shift_shots/lane2" "$LOCAL_ROOT/E11_shift_shots/lane3" \
    --output-dir "$OUTPUT"
.venv/bin/python scripts/plot_shift_shots.py \
    --series "RSG3=$OUTPUT::e11_rsg3_s*" \
    --series "RSG3=$ROOT/results/v7_evidence/E10_dsba3/e10_rsg3" \
    --series "SuperAD-DINOv3=$OUTPUT::e11_superad_dinov3_s*" \
    --series "SuperAD-DINOv3=$ROOT/results/v7_evidence/E10_dsba3/e10_superad_dinov3" \
    --series "SuperAD=$OUTPUT::e11_superad_s*" \
    --series "SuperAD=$ROOT/results/v7_evidence/E10_dsba3/e10_superad" \
    --series "PatchCore=$OUTPUT::e11_patchcore_s*" \
    --series "PatchCore=$ROOT/results/v7_evidence/E10_dsba3/e10_patchcore" \
    --output-dir "$OUTPUT/figures"
.venv/bin/python - "$OUTPUT/figures/shift_shots.csv" <<'PY'
import csv
import sys
from pathlib import Path
rows = list(csv.DictReader(Path(sys.argv[1]).open()))
expected = {(method, str(shot), condition)
            for method in ('RSG3', 'SuperAD-DINOv3', 'SuperAD', 'PatchCore')
            for shot in (1, 2, 4, 8, 16)
            for condition in ('shift_1', 'shift_2', 'shift_3')}
actual = {(r['method'], r['shots'], r['condition']) for r in rows}
if len(rows) != len(expected) or actual != expected:
    raise SystemExit(f'Incomplete E11 figure: {len(rows)} rows; missing {sorted(expected-actual)}')
PY
echo "E11 complete and collected: $OUTPUT"
