#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace}"
DATA_ROOT="${DATA_ROOT:-/home/woojun/dataset/mvtec_ad}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_NAME="${RUN_NAME:-flow_latentbank_mvtecad1_shot4_vitb14reg_static_dvt_alpha_20260713_v1}"
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

run_alpha() {
  local gpu="$1"
  local alpha="$2"
  local tag="${alpha//./}"
  local output_root="${RUN_ROOT}/alpha_${tag}"
  local log_path="${RUN_ROOT}/logs/alpha_${tag}.log"

  if [[ -f "${output_root}/run_manifest.json" ]]; then
    echo "[skip] completed alpha ${alpha}: ${output_root}" | tee -a "${log_path}"
    return 0
  fi

  echo "[start] alpha=${alpha} gpu=${gpu}" | tee "${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" python3 scripts/run_flow_tte_mvtec_ad1.py \
    --data-root "${DATA_ROOT}" \
    --output-root "${output_root}" \
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
    --dvt-denoise-mode position_mean \
    --dvt-denoise-alpha "${alpha}" \
    --normality-mode fused \
    --top-percent 0.01 \
    --query-chunk-size 512 \
    --calibration-sample-size 0 \
    --cleanup-maps 2>&1 | tee -a "${log_path}"
  echo "[complete] alpha=${alpha} gpu=${gpu}" | tee -a "${log_path}"
}

(run_alpha 0 0.0 && run_alpha 0 0.5 && run_alpha 0 1.0) &
gpu0_pid=$!
(run_alpha 1 0.25 && run_alpha 1 0.75) &
gpu1_pid=$!

status=0
wait "${gpu0_pid}" || status=1
wait "${gpu1_pid}" || status=1
if [[ "${status}" -ne 0 ]]; then
  echo "[failed] one or more alpha runs failed" >&2
  exit "${status}"
fi

RUN_ROOT="${RUN_ROOT}" python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
summary = {"run_root": str(root), "alphas": {}}
for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
    tag = str(alpha).replace(".", "")
    with (root / f"alpha_{tag}" / "metrics.json").open() as handle:
        summary["alphas"][str(alpha)] = json.load(handle)
with (root / "dvt_alpha_summary.json").open("w") as handle:
    json.dump(summary, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "[complete] ${RUN_NAME}"
