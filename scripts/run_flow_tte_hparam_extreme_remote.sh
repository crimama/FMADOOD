#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_SUFFIX="${RUN_SUFFIX:-20260710_v1}"
WAIT_FOR_TFCTX="${WAIT_FOR_TFCTX:-1}"
LEADERBOARD="${RESULTS_ROOT}/flowtte_hparam_extreme_${RUN_SUFFIX}_leaderboard.tsv"

if [[ "${WAIT_FOR_TFCTX}" == "1" ]]; then
  while pgrep -af "run_flowtte_tfctx_variants" >/dev/null 2>&1; do
    echo "[wait] flowtte_tfctx_variants still running $(date -Iseconds)"
    sleep 60
  done
fi

cd "${FSAD_ROOT}"
export FMAD_DINOV3_OFFLINE="${FMAD_DINOV3_OFFLINE:-1}"
export DATA_ROOT="${DATA_ROOT:-/home/hunim/Volume/DATA/mvtec_ad_2}"
export PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
export FSAD_ROOT
export BACKBONE_MODEL="${BACKBONE_MODEL:-dinov3_vith16plus}"
export FEATURE_LAYERS="${FEATURE_LAYERS:-7,15,23,31}"
export NORMALITY_MODE="${NORMALITY_MODE:-fused}"
export DVT_DENOISE_MODE="${DVT_DENOISE_MODE:-position_mean}"
export SCORE_MODE="${SCORE_MODE:-latent_distance}"
export FLOW_TRANSFORM_MODE="${FLOW_TRANSFORM_MODE:-flow}"
export FLOW_CONDITION_MODE="${FLOW_CONDITION_MODE:-none}"
export CONTEXT_SOURCE="${CONTEXT_SOURCE:-none}"
export FLOW_CONTEXT_SOURCE="${FLOW_CONTEXT_SOURCE:-auto}"
export MEMORY_CONTEXT_SOURCE="${MEMORY_CONTEXT_SOURCE:-auto}"
export CONTEXT_MODE="${CONTEXT_MODE:-none}"
export CONTEXT_WEIGHT="${CONTEXT_WEIGHT:-0.0}"
export CONTEXT_TOP_M="${CONTEXT_TOP_M:-1}"
export CLEANUP_MAPS="${CLEANUP_MAPS:-1}"
export SUPPORT_SELECTION="${SUPPORT_SELECTION:-fixed_json=${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json}"
export FEATURE_FUSION="${FEATURE_FUSION:-layer_norm_mean}"
export BACKBONE_RESOLUTION="${BACKBONE_RESOLUTION:-0}"
export TILE_PATCH_SIZE="${TILE_PATCH_SIZE:-0}"
export TILE_OVERLAP="${TILE_OVERLAP:-0}"
export IMAGE_RESIZE_FACTOR="${IMAGE_RESIZE_FACTOR:-1.0}"
export TRANSFORMER_CONTEXT_MODE="${TRANSFORMER_CONTEXT_MODE:-none}"

summarize_run() {
  local run_name="$1"
  python3 - "${RESULTS_ROOT}/${run_name}" "${run_name}" "${LEADERBOARD}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
run_name = sys.argv[2]
leaderboard = Path(sys.argv[3])
rows = []
for metrics_path in sorted(root.glob("chunks/*/metrics.json")):
    data = json.loads(metrics_path.read_text())
    for obj, values in data.items():
        if isinstance(values, dict) and "seg_AUROC" in values and "seg_F1" in values:
            rows.append((obj, float(values["seg_AUROC"]), float(values["seg_F1"])))
if not rows:
    raise SystemExit(f"no object metrics found for {run_name}")
mean_auroc = sum(row[1] for row in rows) / len(rows)
mean_f1 = sum(row[2] for row in rows) / len(rows)
summary = {
    "run_name": run_name,
    "objects": [
        {"object": obj, "seg_AUROC_0.05": auroc, "seg_F1": f1}
        for obj, auroc, f1 in rows
    ],
    "mean_seg_AUROC_0.05": mean_auroc,
    "mean_seg_F1": mean_f1,
}
(root / "summary_hparam.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
line = f"{run_name}\t{mean_auroc:.9f}\t{mean_f1:.9f}\t{len(rows)}\n"
if not leaderboard.exists():
    leaderboard.write_text("run_name\tmean_seg_AUROC_0.05\tmean_seg_F1\tobjects\n")
existing = leaderboard.read_text()
if f"{run_name}\t" not in existing:
    with leaderboard.open("a", encoding="utf-8") as handle:
        handle.write(line)
print(line, end="")
PY
}

run_variant() {
  local run_name="$1"
  shift
  local complete_path="${RESULTS_ROOT}/${run_name}/remote_run_complete.txt"
  if [[ -f "${complete_path}" ]]; then
    echo "[skip] ${run_name}"
    summarize_run "${run_name}" || true
    return
  fi
  echo "[start] ${run_name} $(date -Iseconds)"
  env RUN_NAME="${run_name}" OUTPUT_ROOT="${RESULTS_ROOT}/${run_name}" "$@" \
    bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"
  summarize_run "${run_name}"
  echo "[done] ${run_name} $(date -Iseconds)"
}

# Reference-adjacent density and calibration search.
run_variant "flowtte_hparam_${RUN_SUFFIX}_dw005" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.05 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_dw010" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.10 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_dw015" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.15 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_dw020" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.20 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_dw035" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.35 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_dw050" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.50 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_cal0" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 CALIBRATION_SAMPLE_SIZE=0
run_variant "flowtte_hparam_${RUN_SUFFIX}_cal8192" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 CALIBRATION_SAMPLE_SIZE=8192

# DVT strength around the current H+ reference.
run_variant "flowtte_hparam_${RUN_SUFFIX}_a075" DVT_ALPHA=0.75 DENSITY_WEIGHT=0.25 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_a125" DVT_ALPHA=1.25 DENSITY_WEIGHT=0.25 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_a150" DVT_ALPHA=1.50 DENSITY_WEIGHT=0.25 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_a200" DVT_ALPHA=2.00 DENSITY_WEIGHT=0.25 CALIBRATION_SAMPLE_SIZE=4096

# Flow geometry and likelihood regularization.
run_variant "flowtte_hparam_${RUN_SUFFIX}_clamp12" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 FLOW_CLAMP=1.2 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_clamp15" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 FLOW_CLAMP=1.5 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_clamp25" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 FLOW_CLAMP=2.5 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_clamp35" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 FLOW_CLAMP=3.5 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_layers1" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 COUPLING_LAYERS=1 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_layers4" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 COUPLING_LAYERS=4 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_tail0" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 TAIL_WEIGHT=0.0 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_tail06" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 TAIL_WEIGHT=0.6 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_tail003" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 TAIL_TOP_K_RATIO=0.03 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_tail010" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 TAIL_TOP_K_RATIO=0.10 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_logdet0001" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 LAMBDA_LOGDET=1e-4 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_logdet001" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 LAMBDA_LOGDET=1e-2 CALIBRATION_SAMPLE_SIZE=4096

# Training and support augmentation.
run_variant "flowtte_hparam_${RUN_SUFFIX}_lr0001" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 FLOW_LR=1e-4 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_lr0005" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 FLOW_LR=5e-4 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_ep5" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 FLOW_EPOCHS=5 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_br095105" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 SUPPORT_BRIGHTNESS_RANGE=0.95,1.05 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_br090110" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 SUPPORT_BRIGHTNESS_RANGE=0.90,1.10 CALIBRATION_SAMPLE_SIZE=4096
run_variant "flowtte_hparam_${RUN_SUFFIX}_br080120" DVT_ALPHA=1.0 DENSITY_WEIGHT=0.25 SUPPORT_BRIGHTNESS_RANGE=0.80,1.20 CALIBRATION_SAMPLE_SIZE=4096

echo "[complete] hparam sweep ${RUN_SUFFIX} $(date -Iseconds)"
