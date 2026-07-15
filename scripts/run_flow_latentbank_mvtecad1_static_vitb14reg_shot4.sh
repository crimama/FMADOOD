#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/MVTecAD}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flow_latentbank_mvtecad1_all15_shot4_vitb14reg_static_20260712_v1}"
OUTPUT_ROOT="${RESULTS_ROOT}/${RUN_NAME}"
GPU="${GPU:-0}"
OBJECTS="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper"

cd "${FSAD_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
export FMAD_DINOV2_OFFLINE="${FMAD_DINOV2_OFFLINE:-1}"

PYTHONPATH="/workspace:${FSAD_ROOT}" CUDA_VISIBLE_DEVICES="${GPU}" \
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
    if len(train) < 4 or not good or not defects:
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

if [[ -f "${OUTPUT_ROOT}/run_manifest.json" ]]; then
  echo "[skip] completed manifest exists: ${OUTPUT_ROOT}"
  exit 0
fi

CUDA_VISIBLE_DEVICES="${GPU}" python3 scripts/run_flow_tte_mvtec_ad1.py \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --project-root /workspace \
  --fsad-root "${FSAD_ROOT}" \
  --objects "${OBJECTS}" \
  --shots 4 \
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
  --cleanup-maps

echo "[complete] ${RUN_NAME}"
