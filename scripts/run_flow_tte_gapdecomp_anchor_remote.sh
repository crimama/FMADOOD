#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-hunim@147.47.39.144}"
REMOTE_PORT="${REMOTE_PORT:-2222}"
CONTAINER="${CONTAINER:-hun_fsad_tta_012}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/fsad_tta}"
REMOTE_REPO_HOST="${REMOTE_REPO_HOST:-/home/hunim/Volume/FMAD-OOD-remote/fsad_tta}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flowtte_gapdecomp_anchor_20260712_v1}"
REMOTE_RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
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
    pgrep -af run_flow_tte_mvtec_ad2.py || true; \
    for f in \"\$ROOT\"/logs/*.log; do [ -f \"\$f\" ] || continue; \
      echo ===\"\$f\"; tail -n 8 \"\$f\"; done; \
    printf \"retained_tiff_count=\"; \
    find \"\$ROOT/chunks\" -type f -name \"*.tiff\" 2>/dev/null | wc -l; \
    cat \"\$ROOT/remote_run_complete.txt\" 2>/dev/null || true'"
}

pull_remote() {
  mkdir -p "${LOCAL_RUN_ROOT}"
  ssh_remote "docker exec '${CONTAINER}' tar -C '${REMOTE_RUN_ROOT}' -cf - ." \
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
    nohup env FLOWTTE_GAPDECOMP_INTERNAL=1 \
      RUN_NAME=\"${RUN_NAME}\" RESULTS_ROOT=\"${RESULTS_ROOT}\" \
      bash \"${REMOTE_REPO}/scripts/run_flow_tte_gapdecomp_anchor_remote.sh\" internal \
      >\"${REMOTE_RUN_ROOT}/controller.log\" 2>&1 </dev/null &
    echo \$! >\"${REMOTE_RUN_ROOT}/controller.pid\"
    echo controller_pid=\$!
    echo controller_log=${REMOTE_RUN_ROOT}/controller.log
  '"
}

run_chunk() {
  local cuda_slot="$1" chunk_name="$2" objects="$3"
  local chunk_root="${REMOTE_RUN_ROOT}/chunks/${chunk_name}"
  CUDA_VISIBLE_DEVICES="${cuda_slot}" python3 \
    "${REMOTE_REPO}/scripts/run_flow_tte_mvtec_ad2.py" \
    --data-root /home/hunim/Volume/DATA/mvtec_ad_2 \
    --project-root /workspace --fsad-root "${REMOTE_REPO}" \
    --output-root "${chunk_root}" --objects "${objects}" \
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
    --score-field-support-score-quantile 0.90 \
    >"${REMOTE_RUN_ROOT}/logs/${chunk_name}.log" 2>&1
}

summarize() {
  python3 - "${REMOTE_RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("chunks/*/metrics.json")):
    for name, values in json.loads(path.read_text()).items():
        if isinstance(values, dict) and "seg_AUROC" in values:
            rows.append((name, float(values["seg_AUROC"]), float(values["seg_F1"])))
if len(rows) != 8:
    raise SystemExit(f"expected 8 object metrics, found {len(rows)}")
observed = {
    "mean_seg_AUROC_0.05": sum(row[1] for row in rows) / len(rows),
    "mean_seg_F1": sum(row[2] for row in rows) / len(rows),
}
reference = {"mean_seg_AUROC_0.05": 0.8374260727573787, "mean_seg_F1": 0.5306351899731461}
deltas = {key: observed[key] - reference[key] for key in observed}
payload = {
    "claim_scope": "MVTec AD2 test_public diagnostic only; retained maps for Phases 1-2",
    "objects": [{"object": n, "seg_AUROC_0.05": a, "seg_F1": f} for n, a, f in rows],
    "observed": observed,
    "reference": reference,
    "delta_vs_reference": deltas,
    "parity_within_1e-3": {key: abs(value) <= 1e-3 for key, value in deltas.items()},
    "anomaly_maps_retained": True,
}
(root / "summary_gapdecomp_anchor.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
}

internal_controller() {
  export FMAD_DINOV3_OFFLINE=1
  mkdir -p "${REMOTE_RUN_ROOT}/chunks" "${REMOTE_RUN_ROOT}/logs"
  echo "$$" >"${REMOTE_RUN_ROOT}/controller.pid"
  local pids=() status=0
  run_chunk 0 gpu0_can_fabric can,fabric & pids+=("$!")
  run_chunk 1 gpu1_fruit_jelly_rice fruit_jelly,rice & pids+=("$!")
  run_chunk 2 gpu2_vial_wallplugs vial,wallplugs & pids+=("$!")
  run_chunk 3 gpu3_sheet_metal_walnuts sheet_metal,walnuts & pids+=("$!")
  for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
  if [[ "${status}" -ne 0 ]]; then exit "${status}"; fi
  summarize
  printf 'run_name=%s\ncleanup_anomaly_maps=false\nanomaly_maps_retained=true\n' \
    "${RUN_NAME}" >"${REMOTE_RUN_ROOT}/remote_run_complete.txt"
}

case "${MODE}" in
  launch) launch_remote ;;
  poll) poll_remote ;;
  pull) pull_remote ;;
  internal)
    [[ "${FLOWTTE_GAPDECOMP_INTERNAL:-0}" == "1" ]] || {
      echo "internal mode is container-only" >&2; exit 2;
    }
    internal_controller
    ;;
  *) echo "usage: $0 [launch|poll|pull]" >&2; exit 2 ;;
esac
