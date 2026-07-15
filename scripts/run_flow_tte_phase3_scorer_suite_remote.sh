#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-hunim@147.47.39.144}"
REMOTE_PORT="${REMOTE_PORT:-2222}"
CONTAINER="${CONTAINER:-hun_fsad_tta_012}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/fsad_tta}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flowtte_phase3_scorer_suite_20260712_v1}"
REMOTE_RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
ANCHOR_ROOT="${ANCHOR_ROOT:-${RESULTS_ROOT}/flowtte_gapdecomp_anchor_20260712_v1}"
LOCAL_RUN_ROOT="${LOCAL_RUN_ROOT:-${ROOT_DIR}/results/remote_runs/dsba3/${RUN_NAME}}"
MODE="${1:-launch}"

ssh_remote() {
  ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "$@"
}

sync_repo() {
  tar -C "${ROOT_DIR}" -cf - \
    --exclude=.git --exclude=.omx --exclude=.pytest_cache \
    --exclude=.ruff_cache --exclude=.venv --exclude=__pycache__ \
    --exclude=results --exclude=results_remote --exclude=PaperWorks \
    . | ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" \
    "docker exec -i '${CONTAINER}' bash -lc 'mkdir -p \"${REMOTE_REPO}\" && tar -C \"${REMOTE_REPO}\" -xf -'"
}

poll_remote() {
  ssh_remote "docker exec '${CONTAINER}' bash -lc 'ROOT=\"${REMOTE_RUN_ROOT}\"; \
    printf \"controller_pid=\"; cat \"\$ROOT/controller.pid\" 2>/dev/null || true; \
    for f in \"\$ROOT\"/logs/*.pid; do [ -f \"\$f\" ] || continue; \
      pid=\$(cat \"\$f\"); if kill -0 \"\$pid\" 2>/dev/null; then state=running; else state=stopped; fi; \
      echo \"chunk=\$(basename \"\$f\" .pid) pid=\$pid state=\$state\"; done; \
    printf \"completed_objects=\"; find \"\$ROOT/chunks\" -path \"*/objects/*.json\" -type f 2>/dev/null | wc -l; \
    grep -h -m1 \"first_image_scored\" \"\$ROOT\"/logs/*.log 2>/dev/null || true; \
    cat \"\$ROOT/parity_first_object.json\" 2>/dev/null || true; \
    for f in \"\$ROOT\"/logs/*.log; do [ -f \"\$f\" ] || continue; \
      echo ===\"\$f\"; tail -n 8 \"\$f\"; done; \
    cat \"\$ROOT/remote_run_complete.txt\" 2>/dev/null || true'"
}

pull_remote() {
  mkdir -p "${LOCAL_RUN_ROOT}"
  ssh_remote "docker exec '${CONTAINER}' tar \
    --exclude='*.tiff' --exclude='*.npy' --exclude='*.npz' --exclude='*.pt' \
    -C '${REMOTE_RUN_ROOT}' -cf - ." \
    | tar -C "${LOCAL_RUN_ROOT}" -xf -
}

launch_remote() {
  sync_repo
  ssh_remote "docker exec '${CONTAINER}' bash -lc '
    set -e
    mkdir -p \"${REMOTE_RUN_ROOT}\"
    if [ -s \"${REMOTE_RUN_ROOT}/controller.pid\" ] && \
       kill -0 \"\$(cat \"${REMOTE_RUN_ROOT}/controller.pid\")\" 2>/dev/null; then
      echo controller_already_running
      exit 0
    fi
    if [ -e \"${REMOTE_RUN_ROOT}/remote_run_complete.txt\" ]; then
      echo completed_run_already_exists
      exit 0
    fi
    if [ -e \"${REMOTE_RUN_ROOT}/parity_failure.txt\" ]; then
      for pid_file in \"${REMOTE_RUN_ROOT}\"/logs/*.pid; do
        [ -f \"\$pid_file\" ] || continue
        if kill -0 \"\$(cat \"\$pid_file\")\" 2>/dev/null; then
          echo refusing_to_archive_failed_run_with_live_chunk >&2
          exit 1
        fi
      done
      failed_archive=\"${REMOTE_RUN_ROOT}.parity_failed.\$(date -u +%Y%m%dT%H%M%SZ)\"
      mv \"${REMOTE_RUN_ROOT}\" \"\$failed_archive\"
      mkdir -p \"${REMOTE_RUN_ROOT}\"
      echo archived_failed_run=\$failed_archive
    fi
    nohup env FLOWTTE_PHASE3_INTERNAL=1 \
      RUN_NAME=\"${RUN_NAME}\" RESULTS_ROOT=\"${RESULTS_ROOT}\" \
      ANCHOR_ROOT=\"${ANCHOR_ROOT}\" \
      bash \"${REMOTE_REPO}/scripts/run_flow_tte_phase3_scorer_suite_remote.sh\" internal \
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
    "${REMOTE_REPO}/scripts/run_flow_tte_mvtec_ad2.py" \
    --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
    --project-root /workspace --fsad-root "${REMOTE_REPO}" \
    --output-root "${chunk_root}" --objects "${objects}" \
    --multi-scorer-eval --multi-scorer-output "${chunk_root}" \
    --shots 16 --seed 0 --device cuda \
    --flow-epochs 3 --coupling-layers 2 --hidden-multiplier 1 \
    --flow-lr 2e-4 --flow-clamp 1.9 --tail-weight 0.3 \
    --tail-top-k-ratio 0.05 --lambda-logdet 2e-2 \
    --density-quantile 0.90 --expansion-budget 1.0 \
    --distance-weight 1.0 --density-weight 0.25 \
    --score-mode latent_distance --residual-weight 0.25 \
    --top-percent 0.01 --query-chunk-size 512 \
    --pro-integration-limit 0.05 --rgb-guide none --backbone-model dinov3_vith16plus \
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

gate_first_object() {
  local first_object_json="$1"
  python3 - "${REMOTE_RUN_ROOT}" "${ANCHOR_ROOT}" "${first_object_json}" <<'PY'
import json
import sys
from pathlib import Path

run_root, anchor_root, object_path = map(Path, sys.argv[1:])
fallback = {
    "can": 0.000710, "fabric": 0.697949, "fruit_jelly": 0.476761,
    "rice": 0.712533, "vial": 0.439581, "wallplugs": 0.665702,
    "sheet_metal": 0.516126, "walnuts": 0.735719,
}
references = dict(fallback)
reference_source = "prompt_rounded_fallback"
anchor_values = {}
anchor_pauroc_values = {}
for path in sorted(anchor_root.glob("chunks/*/metrics.json")):
    payload = json.loads(path.read_text())
    for name, values in payload.items():
        if isinstance(values, dict) and "seg_F1" in values:
            anchor_values[name] = float(values["seg_F1"])
        if isinstance(values, dict) and "seg_AUROC" in values:
            anchor_pauroc_values[name] = float(values["seg_AUROC"])
if anchor_values:
    references.update(anchor_values)
    reference_source = "anchor_metrics_json_exact"

payload = json.loads(object_path.read_text())
name = str(payload["object"])
observed = float(payload["variants"]["raw_1nn"]["pooled_oracle_f1_float16"])
reference = references[name]
delta = observed - reference
result = {
    "object": name,
    "object_artifact": str(object_path),
    "observed_raw_1nn_f1": observed,
    "reference_anchor_f1": reference,
    "delta": delta,
    "tolerance": 1e-3,
    "pass": abs(delta) <= 1e-3,
    "reference_source": reference_source,
}
(run_root / "parity_first_object.json").write_text(json.dumps(result, indent=2) + "\n")
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
objects = {}
for path in sorted(root.glob("chunks/*/objects/*.json")):
    payload = json.loads(path.read_text())
    name = str(payload["object"])
    if name in objects:
        raise SystemExit(f"duplicate object artifact: {name}")
    objects[name] = payload
expected_objects = {
    "can", "fabric", "fruit_jelly", "rice", "vial", "wallplugs",
    "sheet_metal", "walnuts",
}
if set(objects) != expected_objects:
    raise SystemExit(f"object coverage mismatch: {sorted(objects)}")

expected_variants = {
    "raw_1nn", "density_normalized_1nn_k3", "density_normalized_1nn_k5",
    "density_normalized_1nn_k10", "density_normalized_1nn_k20", "knn_mean_k5",
    "knn_quantile_k10_q0.5", "shrinkage_mahalanobis", "global_pca_residual",
}
for name, payload in objects.items():
    if set(payload["variants"]) != expected_variants:
        raise SystemExit(f"variant coverage mismatch for {name}")

metric_keys = (
    "pooled_oracle_f1_float16", "pooled_pixel_ap_float16",
    "pooled_pauroc_0.05_float16", "tpr_at_fpr_1e-4",
    "normal_image_pooled_fpr_at_oracle",
    "good_pixel_pooled_exceedance_rate_at_oracle",
    "gt_component_recall_at_oracle", "oracle_threshold_float16",
    "good_pixel_p99_9",
)

def finite_mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else float("nan")

per_variant = root / "per_variant"
per_variant.mkdir(exist_ok=True)
summaries = {}
for variant in sorted(expected_variants):
    per_object = {name: objects[name]["variants"][variant] for name in sorted(objects)}
    means = {
        key: finite_mean([metrics[key] for metrics in per_object.values()])
        for key in metric_keys if all(key in metrics for metrics in per_object.values())
    }
    summaries[variant] = {"variant": variant, "means": means, "objects": per_object}
    (per_variant / f"{variant}.json").write_text(
        json.dumps(summaries[variant], indent=2, sort_keys=True) + "\n"
    )

columns = [
    "variant", "mean_f1", "mean_ap", "mean_pauroc_0.05",
    "mean_tpr_at_fpr_1e-4", "vial_f1", "fruit_jelly_f1", "wallplugs_f1",
]
lines = ["\t".join(columns)]
for variant in sorted(expected_variants):
    summary = summaries[variant]
    means = summary["means"]
    per_object = summary["objects"]
    values = [
        variant,
        means["pooled_oracle_f1_float16"],
        means["pooled_pixel_ap_float16"],
        means["pooled_pauroc_0.05_float16"],
        means["tpr_at_fpr_1e-4"],
        per_object["vial"]["pooled_oracle_f1_float16"],
        per_object["fruit_jelly"]["pooled_oracle_f1_float16"],
        per_object["wallplugs"]["pooled_oracle_f1_float16"],
    ]
    lines.append("\t".join([values[0], *(f"{float(value):.10g}" for value in values[1:])]))
(root / "leaderboard.tsv").write_text("\n".join(lines) + "\n")

fallback = {
    "can": 0.000710, "fabric": 0.697949, "fruit_jelly": 0.476761,
    "rice": 0.712533, "vial": 0.439581, "wallplugs": 0.665702,
    "sheet_metal": 0.516126, "walnuts": 0.735719,
}
references = dict(fallback)
reference_source = "prompt_rounded_fallback"
anchor_values = {}
anchor_pauroc_values = {}
for path in sorted(anchor_root.glob("chunks/*/metrics.json")):
    payload = json.loads(path.read_text())
    for name, values in payload.items():
        if isinstance(values, dict) and "seg_F1" in values:
            anchor_values[name] = float(values["seg_F1"])
        if isinstance(values, dict) and "seg_AUROC" in values:
            anchor_pauroc_values[name] = float(values["seg_AUROC"])
if anchor_values:
    references.update(anchor_values)
    reference_source = "anchor_metrics_json_exact"
raw = summaries["raw_1nn"]
raw_objects = summaries["raw_1nn"]["objects"]
parity_objects = {
    name: {
        "observed": float(raw_objects[name]["pooled_oracle_f1_float16"]),
        "reference": references[name],
        "delta": float(raw_objects[name]["pooled_oracle_f1_float16"]) - references[name],
        "pass": abs(float(raw_objects[name]["pooled_oracle_f1_float16"]) - references[name]) <= 1e-3,
    }
    for name in sorted(objects)
}
observed_mean_f1 = float(raw["means"]["pooled_oracle_f1_float16"])
observed_mean_pauroc = float(raw["means"]["pooled_pauroc_0.05_float16"])
reference_mean_f1 = (
    finite_mean(anchor_values.values()) if len(anchor_values) == 8 else 0.530635
)
reference_mean_pauroc = (
    finite_mean(anchor_pauroc_values.values()) if len(anchor_pauroc_values) == 8 else 0.8374
)
mean_checks = {
    "f1": {
        "observed": observed_mean_f1,
        "reference": reference_mean_f1,
        "delta": observed_mean_f1 - reference_mean_f1,
        "pass": abs(observed_mean_f1 - reference_mean_f1) <= 1e-3,
    },
    "pauroc_0.05": {
        "observed": observed_mean_pauroc,
        "reference": reference_mean_pauroc,
        "delta": observed_mean_pauroc - reference_mean_pauroc,
        "pass": abs(observed_mean_pauroc - reference_mean_pauroc) <= 1e-3,
    },
}
parity = {
    "tolerance": 1e-3,
    "reference_source": reference_source,
    "objects": parity_objects,
    "mean_checks": mean_checks,
    "pass": (
        all(item["pass"] for item in parity_objects.values())
        and all(item["pass"] for item in mean_checks.values())
    ),
}
(root / "parity_all_objects.json").write_text(json.dumps(parity, indent=2) + "\n")
if not parity["pass"]:
    raise SystemExit("raw_1nn all-object parity failed")

gate_rows = []
for variant in sorted(expected_variants - {"raw_1nn"}):
    candidate = summaries[variant]
    candidate_objects = candidate["objects"]
    floor_violations = [
        name for name in sorted(objects)
        if (
            float(raw["objects"][name]["pooled_oracle_f1_float16"])
            - float(candidate_objects[name]["pooled_oracle_f1_float16"]) > 0.02
            or float(raw["objects"][name]["pooled_pixel_ap_float16"])
            - float(candidate_objects[name]["pooled_pixel_ap_float16"]) > 0.02
        )
    ]
    checks = {
        "ap_plus_0.01_and_mean_f1_plus_0.015": (
            candidate["means"]["pooled_pixel_ap_float16"] >= raw["means"]["pooled_pixel_ap_float16"] + 0.01
            and candidate["means"]["pooled_oracle_f1_float16"] >= raw["means"]["pooled_oracle_f1_float16"] + 0.015
        ),
        "tpr_at_fpr_1e-4_improved": candidate["means"]["tpr_at_fpr_1e-4"] > raw["means"]["tpr_at_fpr_1e-4"],
        "normal_tail_fpr_not_worse": candidate["means"]["normal_image_pooled_fpr_at_oracle"] <= raw["means"]["normal_image_pooled_fpr_at_oracle"],
        "vial_or_fruit_jelly_f1_plus_0.03": any(
            float(candidate_objects[name]["pooled_oracle_f1_float16"])
            >= float(raw["objects"][name]["pooled_oracle_f1_float16"]) + 0.03
            for name in ("vial", "fruit_jelly")
        ),
        "no_object_floor_violation": not floor_violations,
    }
    gate_rows.append({"variant": variant, "checks": checks, "floor_violations": floor_violations})
(root / "keep_gate.json").write_text(json.dumps(gate_rows, indent=2) + "\n")
gate_lines = ["variant\tgate\tpass\tdetail"]
for row in gate_rows:
    for gate, passed in row["checks"].items():
        detail = ",".join(row["floor_violations"]) if gate == "no_object_floor_violation" else ""
        gate_lines.append(f"{row['variant']}\t{gate}\t{'PASS' if passed else 'FAIL'}\t{detail}")
(root / "keep_gate.tsv").write_text("\n".join(gate_lines) + "\n")

metadata = {
    "claim_scope": "AD2-public-shadow-diagnostic",
    "anomaly_map_tiffs_written": False,
    "class_agnostic_variant_parameters": True,
    "raw_1nn_anchor_parity": parity,
    "calibration_deviations": {
        "shrinkage_mahalanobis": "Support calibration uses in-sample support distances; optimistic bias is possible.",
        "global_pca_residual": "Support calibration uses in-sample support distances; optimistic bias is possible.",
    },
    "object_metadata": {
        name: {key: value for key, value in payload.items() if key not in {"variants"}}
        for name, payload in sorted(objects.items())
    },
}
(root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
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

  run_chunk 0 gpu0_can_fabric can,fabric \
    >"${REMOTE_RUN_ROOT}/logs/gpu0_can_fabric.log" 2>&1 & pids+=("$!")
  echo "${pids[-1]}" >"${REMOTE_RUN_ROOT}/logs/gpu0_can_fabric.pid"
  run_chunk 1 gpu1_fruit_jelly_rice fruit_jelly,rice \
    >"${REMOTE_RUN_ROOT}/logs/gpu1_fruit_jelly_rice.log" 2>&1 & pids+=("$!")
  echo "${pids[-1]}" >"${REMOTE_RUN_ROOT}/logs/gpu1_fruit_jelly_rice.pid"
  run_chunk 2 gpu2_vial_wallplugs vial,wallplugs \
    >"${REMOTE_RUN_ROOT}/logs/gpu2_vial_wallplugs.log" 2>&1 & pids+=("$!")
  echo "${pids[-1]}" >"${REMOTE_RUN_ROOT}/logs/gpu2_vial_wallplugs.pid"
  run_chunk 3 gpu3_sheet_metal_walnuts sheet_metal,walnuts \
    >"${REMOTE_RUN_ROOT}/logs/gpu3_sheet_metal_walnuts.log" 2>&1 & pids+=("$!")
  echo "${pids[-1]}" >"${REMOTE_RUN_ROOT}/logs/gpu3_sheet_metal_walnuts.pid"
  local pid
  for pid in "${pids[@]}"; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "phase3 chunk failed during startup: pid=${pid}" >&2
      exit 1
    fi
  done
  printf 'all_four_chunks_started=true\npids=%s\n' "${pids[*]}" \
    >"${REMOTE_RUN_ROOT}/chunks_started.txt"

  local first_object_json=""
  while [[ -z "${first_object_json}" ]]; do
    first_object_json="$(find "${REMOTE_RUN_ROOT}/chunks" -path '*/objects/*.json' -type f \
      -printf '%T@ %p\n' 2>/dev/null | sort -n | head -n 1 | cut -d' ' -f2- || true)"
    if [[ -n "${first_object_json}" ]]; then break; fi
    local any_alive=0
    for pid in "${pids[@]}"; do
      if kill -0 "${pid}" 2>/dev/null; then any_alive=1; break; fi
    done
    if [[ "${any_alive}" -eq 0 ]]; then
      echo "all chunks exited before producing an object artifact" >&2
      exit 1
    fi
    sleep 5
  done
  gate_first_object "${first_object_json}" || parity_status=$?
  if [[ "${parity_status}" -ne 0 ]]; then
    printf 'parity_pass=false\nexit_code=%s\n' "${parity_status}" \
      >"${REMOTE_RUN_ROOT}/parity_failure.txt"
    cleanup_children
    trap - EXIT INT TERM
    exit "${parity_status}"
  fi

  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  trap - EXIT INT TERM
  if [[ "${status}" -ne 0 ]]; then exit "${status}"; fi
  summarize
  printf 'run_name=%s\nclaim_scope=AD2-public-shadow-diagnostic\nscorer_suite=true\nanomaly_map_tiffs_written=false\nobject_count=8\nvariant_count=9\n' \
    "${RUN_NAME}" >"${REMOTE_RUN_ROOT}/remote_run_complete.txt"
}

case "${MODE}" in
  launch) launch_remote ;;
  poll) poll_remote ;;
  pull) pull_remote ;;
  internal)
    [[ "${FLOWTTE_PHASE3_INTERNAL:-0}" == "1" ]] || {
      echo "internal mode is container-only" >&2; exit 2;
    }
    internal_controller
    ;;
  *) echo "usage: $0 [launch|poll|pull]" >&2; exit 2 ;;
esac
