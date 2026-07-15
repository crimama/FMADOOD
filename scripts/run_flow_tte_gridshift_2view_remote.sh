#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${FLOWTTE_GRIDSHIFT_INTERNAL:-0}" != "1" ]]; then
  PRESET="${REMOTE_PRESET:-${ROOT_DIR}/configs/remote/dsba3.env}"
  if [[ -f "${PRESET}" ]]; then
    # shellcheck disable=SC1090
    source "${PRESET}"
  fi
fi
if [[ "${FLOWTTE_GRIDSHIFT_INTERNAL:-0}" == "1" ]]; then
  REMOTE_HOST="${REMOTE_HOST:-}"
else
  REMOTE_HOST="${REMOTE_HOST:?set REMOTE_HOST or provide REMOTE_PRESET}"
fi
REMOTE_PORT="${REMOTE_PORT:-2222}"
CONTAINER="${CONTAINER_NAME:-${CONTAINER:-hun_fsad_tta_012}}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/fsad_tta}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
DATA_ROOT="${DATA_ROOT:-${REMOTE_DATA_ROOT:-/home/hunim/Volume/DATA}/mvtec_ad_2}"
RUN_NAME="${RUN_NAME:-flowtte_gridshift_2view_20260713_v1}"
REMOTE_RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${RESULTS_ROOT}/flowtte_gapdecomp_anchor_20260712_v1}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-${ROOT_DIR}/results/remote_runs/dsba3/${RUN_NAME}}"
MODE="${1:-launch}"

ssh_remote() { ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "$@"; }

sync_repo() {
  tar -C "${ROOT_DIR}" -cf - \
    --exclude=.git --exclude=.omx --exclude=.pytest_cache --exclude=.ruff_cache \
    --exclude=.venv --exclude=__pycache__ --exclude=results --exclude=results_remote \
    --exclude=PaperWorks --exclude=configs/remote . | ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" \
    "docker exec -i '${CONTAINER}' bash -lc 'mkdir -p \"${REMOTE_REPO}\" && tar -C \"${REMOTE_REPO}\" -xf -'"
}

poll_remote() {
  ssh_remote "docker exec '${CONTAINER}' bash -lc 'ROOT=\"${REMOTE_RUN_ROOT}\"; \
    printf \"controller_pid=\"; cat \"\$ROOT/controller.pid\" 2>/dev/null || true; \
    for f in \"\$ROOT\"/logs/*.pid; do [ -f \"\$f\" ] || continue; \
      pid=\$(cat \"\$f\"); if kill -0 \"\$pid\" 2>/dev/null; then state=running; else state=stopped; fi; \
      echo \"chunk=\$(basename \"\$f\" .pid) pid=\$pid state=\$state\"; done; \
    printf \"completed_objects=\"; find \"\$ROOT/chunks\" -path \"*/objects/*.json\" -type f 2>/dev/null | wc -l; \
    cat \"\$ROOT/parity_first_object.json\" 2>/dev/null || true; \
    cat \"\$ROOT/parity_all_objects.json\" 2>/dev/null || true; \
    for f in \"\$ROOT\"/logs/*.log; do [ -f \"\$f\" ] || continue; \
      echo ===\"\$f\"; tail -n 8 \"\$f\"; done; \
    [ ! -f \"\$ROOT/leaderboard.tsv\" ] || { echo ===leaderboard; cat \"\$ROOT/leaderboard.tsv\"; }; \
    [ ! -f \"\$ROOT/keep_gate.tsv\" ] || { echo ===keep_gate; cat \"\$ROOT/keep_gate.tsv\"; }; \
    [ ! -f \"\$ROOT/phase_drift.tsv\" ] || { echo ===phase_drift; cat \"\$ROOT/phase_drift.tsv\"; }; \
    cat \"\$ROOT/remote_run_complete.txt\" 2>/dev/null || true'"
}

pull_remote() {
  mkdir -p "${LOCAL_RUN_ROOT}"
  ssh_remote "docker exec '${CONTAINER}' tar \
    --exclude='*.tiff' --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
    -C '${REMOTE_RUN_ROOT}' -cf - ." | tar -C "${LOCAL_RUN_ROOT}" -xf -
}

launch_remote() {
  sync_repo
  ssh_remote "docker exec '${CONTAINER}' bash -lc '
    set -e
    mkdir -p \"${REMOTE_RUN_ROOT}\"
    if [ -e \"${REMOTE_RUN_ROOT}/remote_run_complete.txt\" ]; then
      echo completed_run_already_exists
      exit 0
    fi
    if [ -s \"${REMOTE_RUN_ROOT}/controller.pid\" ] && \
       kill -0 \"\$(cat \"${REMOTE_RUN_ROOT}/controller.pid\")\" 2>/dev/null; then
      echo controller_already_running
      exit 0
    fi
    if [ -e \"${REMOTE_RUN_ROOT}/parity_failure.txt\" ] || \
       find \"${REMOTE_RUN_ROOT}/chunks\" -maxdepth 2 -name parity_failure.json \
         -type f -print -quit 2>/dev/null | grep -q .; then
      echo parity_failed_run_requires_explicit_new_RUN_NAME >&2
      exit 1
    fi
    nohup env FLOWTTE_GRIDSHIFT_INTERNAL=1 RUN_NAME=\"${RUN_NAME}\" \
      RESULTS_ROOT=\"${RESULTS_ROOT}\" ANCHOR_ROOT=\"${ANCHOR_ROOT}\" \
      DATA_ROOT=\"${DATA_ROOT}\" \
      bash \"${REMOTE_REPO}/scripts/run_flow_tte_gridshift_2view_remote.sh\" internal \
      >\"${REMOTE_RUN_ROOT}/controller.log\" 2>&1 </dev/null &
    echo \$! >\"${REMOTE_RUN_ROOT}/controller.pid\"
    echo controller_pid=\$!
    echo controller_log=${REMOTE_RUN_ROOT}/controller.log
  '"
}

run_chunk() {
  local cuda_slot="$1" chunk_name="$2" objects="$3"
  local chunk_root="${REMOTE_RUN_ROOT}/chunks/${chunk_name}"
  exec env CUDA_VISIBLE_DEVICES="${cuda_slot}" python3 \
    "${REMOTE_REPO}/scripts/run_flow_tte_gridshift_2view.py" \
    --data-root "${DATA_ROOT}" \
    --project-root /workspace --fsad-root "${REMOTE_REPO}" \
    --output-root "${chunk_root}" --objects "${objects}" \
    --anchor-root "${ANCHOR_ROOT}" --grid-shift-pixels 8 \
    --shots 16 --seed 0 --device cuda \
    --flow-epochs 3 --coupling-layers 2 --hidden-multiplier 1 \
    --flow-lr 2e-4 --flow-clamp 1.9 --tail-weight 0.3 \
    --tail-top-k-ratio 0.05 --lambda-logdet 2e-2 \
    --density-quantile 0.90 --expansion-budget 1.0 \
    --distance-weight 1.0 --density-weight 0.25 \
    --score-mode latent_distance --residual-weight 0.25 \
    --top-percent 0.01 --query-chunk-size 512 \
    --pro-integration-limit 0.05 --backbone-model dinov3_vith16plus \
    --backbone-resolution 0 --feature-layers 7,15,23,31 \
    --tile-patch-size 0 --tile-overlap 0 --image-resize-factor 1.0 \
    --support-brightness-range 0.80,1.20 \
    --support-selection "fixed_json=${REMOTE_REPO}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json" \
    --support-transforms identity --feature-fusion layer_norm_mean \
    --normality-mode fused --context-source none \
    --flow-context-source auto --memory-context-source auto \
    --context-mode none --context-weight 0.0 --context-top-m 1 \
    --calibration-sample-size 4096 --flow-condition-mode none \
    --transformer-context-mode none --flow-transform-mode flow \
    --dvt-denoise-mode position_mean --dvt-denoise-alpha 1.0 \
    --score-field-calibration-mode none \
    --score-field-calibration-alpha 1.0 \
    --score-field-position-std-floor 0.25 \
    --score-field-foreground-mode none \
    --score-field-foreground-quantile 0.20 \
    --score-field-background-multiplier 0.50 \
    --score-field-foreground-smooth-kernel 5 \
    --score-field-support-score-quantile 0.90
}

gate_parity() {
  local object_json="$1" output_json="$2"
  python3 - "${ANCHOR_ROOT}" "${object_json}" "${output_json}" <<'PY'
import json
import sys
from pathlib import Path

anchor_root, object_path, output_path = map(Path, sys.argv[1:])
references = {}
for path in sorted(anchor_root.glob("chunks/*/metrics.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, values in payload.items():
        if isinstance(values, dict) and "seg_F1" in values:
            references[str(name)] = float(values["seg_F1"])
if len(references) != 8:
    raise SystemExit(f"expected 8 exact anchor F1 values, found {len(references)}")
payload = json.loads(object_path.read_text(encoding="utf-8"))
name = str(payload["object"])
observed = float(payload["variants"]["view0_only"]["pooled_oracle_f1"])
reference = references[name]
result = {
    "object": name, "object_artifact": str(object_path),
    "observed": observed, "reference": reference,
    "delta": observed - reference, "tolerance": 0.0,
    "pass": observed == reference,
}
output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True), flush=True)
raise SystemExit(0 if result["pass"] else 42)
PY
}

summarize() {
  python3 - "${REMOTE_RUN_ROOT}" "${ANCHOR_ROOT}" <<'PY'
import json
import math
import sys
from pathlib import Path

root, anchor_root = map(Path, sys.argv[1:])
expected_objects = {"can", "fabric", "fruit_jelly", "rice", "vial", "wallplugs", "sheet_metal", "walnuts"}
variants = ("view0_only", "view1_arm_A", "arm_A_mean", "arm_A_max", "arm_C_mean", "arm_C_max")
objects = {}
for path in sorted(root.glob("chunks/*/objects/*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = str(payload["object"])
    if name in objects:
        raise SystemExit(f"duplicate object artifact: {name}")
    if tuple(payload["variants"].keys()) != variants and set(payload["variants"]) != set(variants):
        raise SystemExit(f"variant coverage mismatch for {name}: {sorted(payload['variants'])}")
    objects[name] = payload
if set(objects) != expected_objects:
    raise SystemExit(f"object coverage mismatch: {sorted(objects)}")

references = {}
for path in sorted(anchor_root.glob("chunks/*/metrics.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, values in payload.items():
        if isinstance(values, dict) and "seg_F1" in values:
            references[str(name)] = float(values["seg_F1"])
if set(references) != expected_objects:
    raise SystemExit(f"exact anchor coverage mismatch: {sorted(references)}")
parity_objects = {}
for name in sorted(objects):
    observed = float(objects[name]["variants"]["view0_only"]["pooled_oracle_f1"])
    reference = references[name]
    parity_objects[name] = {
        "observed": observed, "reference": reference,
        "delta": observed - reference, "pass": observed == reference,
    }
parity = {"tolerance": 0.0, "objects": parity_objects,
          "pass": all(row["pass"] for row in parity_objects.values())}
(root / "parity_all_objects.json").write_text(json.dumps(parity, indent=2) + "\n")
if not parity["pass"]:
    raise SystemExit("view0_only all-object exact parity failed")

def mean(values):
    values = [float(value) for value in values]
    return sum(values) / len(values)

def row_metrics(metric):
    boundary = metric["boundary_tolerant_f1_native_px"]
    return {
        "f1": float(metric["pooled_oracle_f1"]),
        "threshold": float(metric["pooled_oracle_threshold"]),
        "pauroc": float(metric["seg_AUROC_0.05"]),
        "component_recall": float(metric["gt_component_recall"]),
        "boundary_t0": float(boundary["0"]["f1"]),
        "boundary_t4": float(boundary["4"]["f1"]),
        "boundary_t8": float(boundary["8"]["f1"]),
        "normal_fpr": float(metric["normal_image_fpr"]["mean_per_image"]),
    }

summary = {}
columns = ("object", "variant", "f1", "threshold", "pauroc_0.05", "component_recall",
           "boundary_t0", "boundary_t4", "boundary_t8", "normal_image_mean_fpr")
lines = ["\t".join(columns)]
for name in sorted(objects):
    for variant in variants:
        row = row_metrics(objects[name]["variants"][variant])
        summary.setdefault(variant, {})[name] = row
        lines.append("\t".join((name, variant, *(f"{row[key]:.10g}" for key in
            ("f1", "threshold", "pauroc", "component_recall", "boundary_t0", "boundary_t4", "boundary_t8", "normal_fpr")))))
means = {}
for variant in variants:
    means[variant] = {key: mean(row[key] for row in summary[variant].values())
                      for key in ("f1", "threshold", "pauroc", "component_recall", "boundary_t0", "boundary_t4", "boundary_t8", "normal_fpr")}
    row = means[variant]
    lines.append("\t".join(("MEAN", variant, *(f"{row[key]:.10g}" for key in
        ("f1", "threshold", "pauroc", "component_recall", "boundary_t0", "boundary_t4", "boundary_t8", "normal_fpr")))))
(root / "leaderboard.tsv").write_text("\n".join(lines) + "\n")

baseline = means["view0_only"]
gate_rows = []
for variant in ("arm_A_mean", "arm_A_max", "arm_C_mean", "arm_C_max"):
    candidate = means[variant]
    per_object_f1_deltas = {
        name: summary[variant][name]["f1"] - summary["view0_only"][name]["f1"]
        for name in sorted(objects)
    }
    observed = {
        "component_recall_delta": candidate["component_recall"] - baseline["component_recall"],
        "boundary_t4_delta": candidate["boundary_t4"] - baseline["boundary_t4"],
        "boundary_t8_delta": candidate["boundary_t8"] - baseline["boundary_t8"],
        "pauroc_loss": baseline["pauroc"] - candidate["pauroc"],
        "normal_fpr_delta": candidate["normal_fpr"] - baseline["normal_fpr"],
        "min_per_object_f1_delta": min(per_object_f1_deltas.values()),
    }
    gate_rows.append({"variant": variant, "observed": observed,
                      "per_object_f1_deltas": per_object_f1_deltas})
(root / "keep_gate.json").write_text(json.dumps(gate_rows, indent=2) + "\n")
gate_lines = ["variant\tcriterion\tobserved\tthreshold"]
gate_specs = (
    ("component_recall_plus_0.05", "component_recall_delta", ">=0.05"),
    ("boundary_t4_or_t8_plus_0.02", None, "max(t4,t8)>=0.02"),
    ("mean_pauroc_loss_le_0.003", "pauroc_loss", "<=0.003"),
    ("normal_fpr_not_worse", "normal_fpr_delta", "<=0"),
    ("per_object_f1_floor_minus_0.02", "min_per_object_f1_delta", ">=-0.02"),
)
for row in gate_rows:
    for criterion, key, threshold in gate_specs:
        value = (max(row["observed"]["boundary_t4_delta"], row["observed"]["boundary_t8_delta"])
                 if key is None else row["observed"][key])
        gate_lines.append(f"{row['variant']}\t{criterion}\t{value:.10g}\t{threshold}")
(root / "keep_gate.tsv").write_text("\n".join(gate_lines) + "\n")

drift_lines = ["object\tgood_pixel_median_view1_minus_view0"]
phase_drift = {}
for name in sorted(objects):
    diagnostic = objects[name].get("diagnostics", {})
    value = diagnostic.get("phase_drift_good_pixel_median_delta")
    if value is None:
        value = objects[name].get("phase_drift_good_pixel_median_delta")
    if value is None or not math.isfinite(float(value)):
        raise SystemExit(f"missing finite phase drift diagnostic for {name}")
    phase_drift[name] = float(value)
    drift_lines.append(f"{name}\t{float(value):.10g}")
(root / "phase_drift.tsv").write_text("\n".join(drift_lines) + "\n")
(root / "summary.json").write_text(json.dumps({
    "variants": summary, "means": means, "parity": parity,
    "keep_gate": gate_rows, "phase_drift": phase_drift,
    "anomaly_map_tiffs_written": False,
}, indent=2, sort_keys=True) + "\n")
PY
}

internal_controller() {
  export FMAD_DINOV3_OFFLINE=1
  mkdir -p "${REMOTE_RUN_ROOT}/chunks" "${REMOTE_RUN_ROOT}/logs"
  echo "$$" >"${REMOTE_RUN_ROOT}/controller.pid"
  local pids=() status=0 parity_status=0
  cleanup_children() {
    local pid
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then kill "${pid}" 2>/dev/null || true; fi
    done
    wait "${pids[@]}" 2>/dev/null || true
  }
  trap cleanup_children EXIT
  trap 'cleanup_children; exit 1' INT TERM

  run_chunk 0 gpu0_can_fabric can,fabric >"${REMOTE_RUN_ROOT}/logs/gpu0_can_fabric.log" 2>&1 & pids+=("$!")
  echo "${pids[-1]}" >"${REMOTE_RUN_ROOT}/logs/gpu0_can_fabric.pid"
  run_chunk 1 gpu1_fruit_jelly_rice fruit_jelly,rice >"${REMOTE_RUN_ROOT}/logs/gpu1_fruit_jelly_rice.log" 2>&1 & pids+=("$!")
  echo "${pids[-1]}" >"${REMOTE_RUN_ROOT}/logs/gpu1_fruit_jelly_rice.pid"
  run_chunk 2 gpu2_vial_wallplugs vial,wallplugs >"${REMOTE_RUN_ROOT}/logs/gpu2_vial_wallplugs.log" 2>&1 & pids+=("$!")
  echo "${pids[-1]}" >"${REMOTE_RUN_ROOT}/logs/gpu2_vial_wallplugs.pid"
  run_chunk 3 gpu3_sheet_metal_walnuts sheet_metal,walnuts >"${REMOTE_RUN_ROOT}/logs/gpu3_sheet_metal_walnuts.log" 2>&1 & pids+=("$!")
  echo "${pids[-1]}" >"${REMOTE_RUN_ROOT}/logs/gpu3_sheet_metal_walnuts.pid"
  printf 'all_four_chunks_started=true\npids=%s\n' "${pids[*]}" >"${REMOTE_RUN_ROOT}/chunks_started.txt"

  local first_object_json=""
  while [[ -z "${first_object_json}" ]]; do
    first_object_json="$(find "${REMOTE_RUN_ROOT}/chunks" -path '*/objects/*.json' -type f \
      -printf '%T@ %p\n' 2>/dev/null | sort -n | head -n 1 | cut -d' ' -f2- || true)"
    [[ -z "${first_object_json}" ]] || break
    local any_alive=0 pid
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then any_alive=1; break; fi
    done
    if [[ "${any_alive}" -eq 0 ]]; then
      echo "all chunks exited before producing an object artifact" >&2
      exit 1
    fi
    sleep 5
  done
  gate_parity "${first_object_json}" "${REMOTE_RUN_ROOT}/parity_first_object.json" || parity_status=$?
  if [[ "${parity_status}" -ne 0 ]]; then
    printf 'parity_pass=false\nexit_code=%s\n' "${parity_status}" >"${REMOTE_RUN_ROOT}/parity_failure.txt"
    cleanup_children
    trap - EXIT INT TERM
    exit "${parity_status}"
  fi

  # Every object runner writes a chunk-local parity sentinel before exiting.
  # Monitor all shards so a later-object mismatch stops the global smoke, not
  # merely its own two-object shard.
  while true; do
    local chunk_parity_failure
    chunk_parity_failure="$(find "${REMOTE_RUN_ROOT}/chunks" -maxdepth 2 \
      -name parity_failure.json -type f -print -quit 2>/dev/null || true)"
    if [[ -n "${chunk_parity_failure}" ]]; then
      printf 'parity_pass=false\nsource=%s\n' "${chunk_parity_failure}" \
        >"${REMOTE_RUN_ROOT}/parity_failure.txt"
      cleanup_children
      trap - EXIT INT TERM
      exit 42
    fi
    local any_alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then any_alive=1; break; fi
    done
    [[ "${any_alive}" -eq 1 ]] || break
    sleep 5
  done

  local pid
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  trap - EXIT INT TERM
  [[ "${status}" -eq 0 ]] || exit "${status}"
  if ! summarize; then
    printf 'parity_or_summary_pass=false\n' >"${REMOTE_RUN_ROOT}/parity_failure.txt"
    exit 42
  fi
  local retained_tiffs
  retained_tiffs="$(find "${REMOTE_RUN_ROOT}/chunks" -type f -name '*.tiff' -print -quit)"
  if [[ -n "${retained_tiffs}" ]]; then
    echo "unexpected retained TIFF: ${retained_tiffs}" >&2
    exit 1
  fi
  printf 'run_name=%s\nobject_count=8\nvariant_count=6\nanomaly_map_tiffs_written=false\nparity_exact=true\n' \
    "${RUN_NAME}" >"${REMOTE_RUN_ROOT}/remote_run_complete.txt"
}

case "${MODE}" in
  launch) launch_remote ;;
  poll) poll_remote ;;
  pull) pull_remote ;;
  internal)
    [[ "${FLOWTTE_GRIDSHIFT_INTERNAL:-0}" == "1" ]] || {
      echo "internal mode is container-only" >&2; exit 2;
    }
    internal_controller
    ;;
  *) echo "usage: $0 [launch|poll|pull]" >&2; exit 2 ;;
esac
