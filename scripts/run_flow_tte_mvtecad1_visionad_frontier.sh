#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/MVTecAD}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flowtte_mvtecad1_visionad_vitb14reg_s1_2_4_8_16_20260712}"
RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
OBJECTS="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"

cd "${FSAD_ROOT}"
mkdir -p "${RUN_ROOT}/logs"
export FMAD_DINOV2_OFFLINE="${FMAD_DINOV2_OFFLINE:-1}"

run_shot() {
  local gpu="$1"
  local shot="$2"
  local output="${RUN_ROOT}/shot_${shot}"
  if [[ -f "${output}/run_manifest.json" ]]; then
    echo "[skip] shot=${shot} manifest exists"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/run_flow_tte_mvtec_ad1.py \
    --data-root "${DATA_ROOT}" \
    --output-root "${output}" \
    --project-root /workspace \
    --fsad-root "${FSAD_ROOT}" \
    --objects "${OBJECTS}" \
    --shots "${shot}" \
    --seed 1 \
    --device cuda \
    --backbone-model dinov2_vitb14_reg \
    --preprocess-recipe visionad_official \
    --image-size 448 \
    --crop-size 392 \
    --feature-layers 2,5,8,11 \
    --feature-fusion visionad_mean_l2 \
    --support-selection visionad_seeded_random \
    --support-selection-seed 1 \
    --support-transforms identity \
    --support-brightness-range 0.8,1.2 \
    --flow-epochs 3 \
    --coupling-layers 2 \
    --hidden-multiplier 1 \
    --flow-lr 2e-4 \
    --flow-clamp 1.9 \
    --flow-transform-mode flow \
    --tail-weight 0.3 \
    --tail-top-k-ratio 0.05 \
    --lambda-logdet 0.02 \
    --density-quantile 0.90 \
    --expansion-budget 1.0 \
    --distance-weight 1.0 \
    --density-weight 0.25 \
    --score-mode latent_distance \
    --dvt-denoise-mode position_mean \
    --dvt-denoise-alpha 1.0 \
    --normality-mode fused \
    --top-percent 0.01 \
    --query-chunk-size 512 \
    --calibration-sample-size 4096 \
    --cleanup-maps \
    >"${RUN_ROOT}/logs/shot_${shot}.log" 2>&1
}

run_gpu0() { run_shot 0 1; run_shot 0 4; run_shot 0 16; }
run_gpu1() { run_shot 1 2; run_shot 1 8; }

run_gpu0 & pid0=$!
run_gpu1 & pid1=$!
status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "[failed] inspect ${RUN_ROOT}/logs" >&2
  exit "${status}"
fi

python3 - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
keys = ("i_AUROC", "i_AUPRC", "p_AUROC", "p_AUPRC", "p_AUPRO")
rows = {}
for shot in (1, 2, 4, 8, 16):
    metrics = json.loads((root / f"shot_{shot}" / "metrics.json").read_text())
    rows[str(shot)] = {key: metrics[key] for key in keys}
payload = {"shots": rows, "metric_order": list(keys)}
(root / "shot_sweep_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
echo "[complete] ${RUN_NAME}"
