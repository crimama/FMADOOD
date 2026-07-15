#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/home/woojun/dataset/mvtec_ad}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flow_latentbank_mvtecad1_all15_static_vitb14reg_s1_2_8_20260713_v1}"
RUN_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
OBJECTS="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"

cd "${FSAD_ROOT}"
mkdir -p "${RUN_ROOT}/logs"
export FMAD_DINOV2_OFFLINE="${FMAD_DINOV2_OFFLINE:-1}"

PYTHONPATH="/workspace:${FSAD_ROOT}" CUDA_VISIBLE_DEVICES=0 \
  DATA_ROOT="${DATA_ROOT}" python3 - <<'PY'
import os
from pathlib import Path

from fmad.backbones.dinov2 import DINOv2Backbone

objects = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
)
root = Path(os.environ["DATA_ROOT"])
for object_name in objects:
    train = list((root / object_name / "train" / "good").glob("*"))
    good = list((root / object_name / "test" / "good").glob("*"))
    defects = [path for path in (root / object_name / "test").glob("*") if path.name != "good"]
    if len(train) < 8 or not good or not defects:
        raise SystemExit(f"incomplete MVTec AD category: {object_name}")

sample = sorted((root / "bottle" / "train" / "good").glob("*"))[0]
backbone = DINOv2Backbone(
    model_name="dinov2_vitb14_reg",
    device="cuda",
    smaller_edge_size=448,
    feature_layers=(2, 5, 8, 11),
)
image, grid = backbone.prepare_image(str(sample))
features = backbone.extract_features(image)
model = backbone._wrapper.model
expected_tokens = grid[0] * grid[1]
if len(features) != 4 or any(feature.shape != (expected_tokens, 768) for feature in features):
    raise SystemExit(f"unexpected feature contract: grid={grid}, shapes={[x.shape for x in features]}")
if model.training or any(parameter.requires_grad for parameter in model.parameters()):
    raise SystemExit("DINOv2 encoder is not frozen in eval mode")
print(
    f"[preflight] categories=15 layers={backbone.feature_layers} grid={grid} "
    f"shapes={[feature.shape for feature in features]} frozen=true"
)
PY

run_shot() {
  local gpu="$1"
  local shot="$2"
  local output_root="${RUN_ROOT}/shot_${shot}"
  local log_path="${RUN_ROOT}/logs/shot_${shot}.log"

  if [[ -f "${output_root}/run_manifest.json" ]]; then
    echo "[skip] completed shot ${shot}: ${output_root}" | tee -a "${log_path}"
    return 0
  fi

  echo "[start] shot=${shot} gpu=${gpu}" | tee "${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/run_flow_tte_mvtec_ad1.py \
    --data-root "${DATA_ROOT}" \
    --output-root "${output_root}" \
    --project-root /workspace \
    --fsad-root "${FSAD_ROOT}" \
    --objects "${OBJECTS}" \
    --shots "${shot}" \
    --seed 0 \
    --device cuda \
    --backbone-model dinov2_vitb14_reg \
    --preprocess-recipe fmad_shorter_edge \
    --image-size 448 \
    --crop-size 448 \
    --feature-layers 2,5,8,11 \
    --feature-fusion layer_norm_mean \
    --support-selection first \
    --support-selection-seed 0 \
    --support-transforms identity \
    --support-brightness-range 1.0,1.0 \
    --flow-epochs 3 \
    --coupling-layers 2 \
    --hidden-multiplier 1 \
    --flow-lr 2e-4 \
    --flow-clamp 1.9 \
    --flow-transform-mode flow \
    --tail-weight 0.3 \
    --tail-top-k-ratio 0.05 \
    --lambda-logdet 1e-3 \
    --density-quantile 0.90 \
    --expansion-budget 1.0 \
    --distance-weight 1.0 \
    --density-weight 0.25 \
    --score-mode latent_distance \
    --dvt-denoise-mode none \
    --normality-mode fused \
    --top-percent 0.01 \
    --query-chunk-size 512 \
    --calibration-sample-size 0 \
    --cleanup-maps 2>&1 | tee -a "${log_path}"
  echo "[complete] shot=${shot} gpu=${gpu}" | tee -a "${log_path}"
}

(run_shot 0 1 && run_shot 0 8) &
gpu0_pid=$!
(run_shot 1 2) &
gpu1_pid=$!

status=0
wait "${gpu0_pid}" || status=1
wait "${gpu1_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "[failed] one or more shot runs failed" >&2
  exit "${status}"
fi

RUN_ROOT="${RUN_ROOT}" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
summary = {"run_root": str(root), "shots": {}}
for shot in (1, 2, 8):
    metrics_path = root / f"shot_{shot}" / "metrics.json"
    with metrics_path.open() as handle:
        summary["shots"][str(shot)] = json.load(handle)
with (root / "shot_sweep_summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[complete] ${RUN_NAME}"
