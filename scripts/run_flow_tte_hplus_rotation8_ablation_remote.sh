#!/usr/bin/env bash
set -euo pipefail

FSAD_ROOT="${FSAD_ROOT:-/workspace/fsad_tta}"
PROJECT_ROOT="${PROJECT_ROOT:-/workspace}"
RESULTS_ROOT="${RESULTS_ROOT:-/workspace/results_remote}"
RUN_SUFFIX="${RUN_SUFFIX:-20260710_v1}"
WAIT_FOR_PHASE3="${WAIT_FOR_PHASE3:-1}"
PHASE3_LOG="${RESULTS_ROOT}/flowtte_hparam_phase3_20260710_v3_controller.log"
GROUP_ROOT="${RESULTS_ROOT}/flowtte_hplus_dvt_superad_rotation8_ablation_${RUN_SUFFIX}"
SUPPORT_MANIFEST="${FSAD_ROOT}/skill_graph/experiments/2026-07-07_flowtte_register_failure_analysis/dinov3_noctx_support_paths.json"
SUPPORT_SELECTION_POLICY="fixed_json=${SUPPORT_MANIFEST}"
ALL_OBJECTS="can,fabric,fruit_jelly,rice,vial,wallplugs,walnuts,sheet_metal"
ROTATION8_TRANSFORMS="superad_rot000,superad_rot045,superad_rot090,superad_rot135,superad_rot180,superad_rot225,superad_rot270,superad_rot315"
SUPPORT_MANIFEST_SHA256="$(sha256sum "${SUPPORT_MANIFEST}" | awk '{print $1}')"
mapfile -t METHOD_BUNDLE_FILES < <(
  {
    find "${FSAD_ROOT}/src/flow_tte" -maxdepth 1 -type f -name '*.py' -print
    find "${FSAD_ROOT}/scripts" -maxdepth 1 -type f -name 'flow_tte_*.py' -print
    find "${PROJECT_ROOT}/fmad/datasets" -maxdepth 1 -type f -name '*.py' -print
    find "${PROJECT_ROOT}/fmad/evaluation" -maxdepth 1 -type f -name '*.py' -print
    printf '%s\n' \
      "${FSAD_ROOT}/scripts/dinov3_backbone.py" \
      "${FSAD_ROOT}/scripts/run_flow_tte_mvtec_ad2.py" \
      "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh" \
      "${FSAD_ROOT}/scripts/run_flow_tte_hplus_rotation8_remote.sh" \
      "${FSAD_ROOT}/scripts/run_flow_tte_hplus_rotation8_ablation_remote.sh" \
      "${PROJECT_ROOT}/fmad/registry.py" \
      "${PROJECT_ROOT}/src/post_eval.py" \
      "${PROJECT_ROOT}/src/utils.py"
  } | sort -u
)
METHOD_BUNDLE_SHA256="$(
  sha256sum "${METHOD_BUNDLE_FILES[@]}" \
    | awk '{print $1}' \
    | sha256sum \
    | awk '{print $1}'
)"
COMMON_ENV=(
  PROJECT_ROOT="${PROJECT_ROOT}"
  BACKBONE_MODEL=dinov3_vith16plus
  FEATURE_LAYERS=7,15,23,31
  DVT_DENOISE_MODE=position_mean
  DVT_ALPHA=1.0
  NORMALITY_MODE=fused
  FEATURE_FUSION=layer_norm_mean
  SUPPORT_SELECTION="${SUPPORT_SELECTION_POLICY}"
  SUPPORT_BRIGHTNESS_RANGE=1.0,1.0
  CALIBRATION_SAMPLE_SIZE=4096
  FLOW_EPOCHS=3
  COUPLING_LAYERS=2
  HIDDEN_MULTIPLIER=1
  FLOW_LR=2e-4
  FLOW_CLAMP=1.9
  FLOW_TRANSFORM_MODE=flow
  TAIL_WEIGHT=0.3
  TAIL_TOP_K_RATIO=0.05
  LAMBDA_LOGDET=1e-3
  TOP_PERCENT=0.01
  QUERY_CHUNK_SIZE=512
  DENSITY_WEIGHT=0.25
  SCORE_MODE=latent_distance
  RESIDUAL_WEIGHT=0.25
  CONTEXT_SOURCE=none
  FLOW_CONTEXT_SOURCE=auto
  MEMORY_CONTEXT_SOURCE=auto
  CONTEXT_MODE=none
  CONTEXT_WEIGHT=0.0
  CONTEXT_TOP_M=1
  FLOW_CONDITION_MODE=none
  TRANSFORMER_CONTEXT_MODE=none
  SCORE_FIELD_CALIBRATION_MODE=none
  SCORE_FIELD_CALIBRATION_ALPHA=1.0
  SCORE_FIELD_POSITION_STD_FLOOR=0.25
  SCORE_FIELD_FOREGROUND_MODE=none
  SCORE_FIELD_FOREGROUND_QUANTILE=0.20
  SCORE_FIELD_BACKGROUND_MULTIPLIER=0.50
  SCORE_FIELD_FOREGROUND_SMOOTH_KERNEL=5
  SCORE_FIELD_SUPPORT_SCORE_QUANTILE=0.90
  CLEANUP_MAPS=1
  FMAD_DINOV3_OFFLINE=1
  METHOD_BUNDLE_SHA256="${METHOD_BUNDLE_SHA256}"
)

if [[ "${WAIT_FOR_PHASE3}" == "1" ]]; then
  until grep -q "\[complete\] hparam phase3 20260710_v3" "${PHASE3_LOG}" 2>/dev/null; do
    if ! pgrep -f '[r]un_flow_tte_hparam_phase3_remote.sh' >/dev/null; then
      echo "[error] phase3 controller exited without its completion marker" >&2
      exit 1
    fi
    echo "[wait] hparam phase3 still running $(date -Iseconds)"
    sleep 60
  done
fi

validate_completed_run() {
  local run_name="$1"
  local expected_objects="$2"
  local expected_transforms="$3"
  python3 - \
    "${RESULTS_ROOT}/${run_name}" \
    "${SUPPORT_SELECTION_POLICY}" \
    "${expected_objects}" \
    "${expected_transforms}" \
    "${SUPPORT_MANIFEST_SHA256}" \
    "${METHOD_BUNDLE_SHA256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
support_policy = sys.argv[2]
expected_objects = set(sys.argv[3].split(","))
expected_transforms = sys.argv[4].split(",")
expected_support_sha256 = sys.argv[5]
expected_method_sha256 = sys.argv[6]
support_path = Path(support_policy[len("fixed_json=") :])
support_bytes = support_path.read_bytes()
observed_support_sha256 = hashlib.sha256(support_bytes).hexdigest()
if observed_support_sha256 != expected_support_sha256:
    raise SystemExit("support manifest changed during the run")
expected_support_paths = json.loads(support_bytes.decode("utf-8"))
manifest_paths = sorted((run_root / "chunks").glob("*/run_manifest.json"))
if not manifest_paths:
    raise SystemExit(f"missing chunk manifests: {run_root}")

observed_objects = set()
for manifest_path in manifest_paths:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_objects.update(manifest["objects"])
    expected = {
        "shots": 16,
        "seed": 0,
        "support_policy": support_policy,
        "support_selection_seed": 0,
        "support_transforms": expected_transforms,
        "effective_support_view_count": 16 * len(expected_transforms),
        "dvt_denoise_mode": "position_mean",
        "dvt_denoise_alpha": 1.0,
        "backbone": "dinov3_vith16plus layer mean [7, 15, 23, 31]",
        "support_brightness_range": [1.0, 1.0],
        "feature_fusion": "layer_norm_mean",
        "normality_mode": "fused",
        "residual_weight": 0.25,
        "flow_epochs": 3,
        "coupling_layers": 2,
        "hidden_multiplier": 1,
        "flow_lr": 0.0002,
        "flow_clamp": 1.9,
        "flow_transform_mode": "flow",
        "tail_weight": 0.3,
        "tail_top_k_ratio": 0.05,
        "lambda_logdet": 0.001,
        "density_quantile": 0.9,
        "expansion_budget": 1.0,
        "distance_weight": 1.0,
        "density_weight": 0.25,
        "score_mode": "latent_distance",
        "top_percent": 0.01,
        "query_chunk_size": 512,
        "calibration_sample_size": 4096,
        "use_squared_distance": False,
        "flow_condition_mode": "none",
        "transformer_context_mode": "none",
        "context_source": "none",
        "flow_context_source": "auto",
        "memory_context_source": "auto",
        "resolved_flow_context_source": "none",
        "resolved_memory_context_source": "none",
        "context_mode": "none",
        "context_weight": 0.0,
        "context_top_m": 1,
        "score_field_calibration_mode": "none",
        "score_field_calibration_alpha": 1.0,
        "score_field_position_std_floor": 0.25,
        "score_field_foreground_mode": "none",
        "score_field_foreground_quantile": 0.2,
        "score_field_background_multiplier": 0.5,
        "score_field_foreground_smooth_kernel": 5,
        "score_field_support_score_quantile": 0.9,
        "cleanup_maps": True,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise SystemExit(f"manifest mismatch {manifest_path}: {mismatches}")
    for diagnostic in manifest["object_diagnostics"]:
        object_name = diagnostic["object_name"]
        if diagnostic["selected_support_paths"] != expected_support_paths[object_name]:
            raise SystemExit(f"support-path mismatch {manifest_path}: {object_name}")
    if not (manifest_path.parent / "cleanup_evidence.txt").is_file():
        raise SystemExit(f"missing cleanup evidence: {manifest_path.parent}")
    if (manifest_path.parent / "anomaly_maps").exists():
        raise SystemExit(f"anomaly maps were not cleaned: {manifest_path.parent}")

if observed_objects != expected_objects:
    raise SystemExit(
        f"object mismatch {run_root}: observed={sorted(observed_objects)} "
        f"expected={sorted(expected_objects)}"
    )
root_marker = {}
for line in (run_root / "remote_run_complete.txt").read_text(encoding="utf-8").splitlines():
    key, value = line.split("=", 1)
    root_marker[key] = value
if root_marker.get("method_bundle_sha256") != expected_method_sha256:
    raise SystemExit(f"method bundle mismatch: {run_root}")
print(f"[validated] {run_root.name} objects={len(observed_objects)} views={16 * len(expected_transforms)}")
PY
}

run_variant() {
  local run_name="$1"
  local support_transforms="$2"
  local complete_path="${RESULTS_ROOT}/${run_name}/remote_run_complete.txt"
  if [[ -f "${complete_path}" ]]; then
    validate_completed_run "${run_name}" "${ALL_OBJECTS}" "${support_transforms}"
    echo "[skip] ${run_name}"
    return
  fi
  echo "[start] ${run_name} $(date -Iseconds)"
  env \
    "${COMMON_ENV[@]}" \
    RUN_NAME="${run_name}" \
    OUTPUT_ROOT="${RESULTS_ROOT}/${run_name}" \
    SUPPORT_TRANSFORMS="${support_transforms}" \
    SMOKE_OBJECT="" \
    bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"
  validate_completed_run "${run_name}" "${ALL_OBJECTS}" "${support_transforms}"
  echo "[done] ${run_name} $(date -Iseconds)"
}

mkdir -p "${GROUP_ROOT}"

SMOKE_NAME="flowtte_hplus_dvt_superad_rotation8_can_smoke_${RUN_SUFFIX}"
if [[ -f "${RESULTS_ROOT}/${SMOKE_NAME}/remote_run_complete.txt" ]]; then
  validate_completed_run "${SMOKE_NAME}" "can" "${ROTATION8_TRANSFORMS}"
  echo "[skip] ${SMOKE_NAME}"
else
  echo "[start] ${SMOKE_NAME} $(date -Iseconds)"
  env \
    "${COMMON_ENV[@]}" \
    RUN_NAME="${SMOKE_NAME}" \
    OUTPUT_ROOT="${RESULTS_ROOT}/${SMOKE_NAME}" \
    SUPPORT_TRANSFORMS="${ROTATION8_TRANSFORMS}" \
    SMOKE_OBJECT="can" \
    SMOKE_CUDA_SLOT="0" \
    bash "${FSAD_ROOT}/scripts/run_flow_tte_dvt_denoising_all8_remote.sh"
  validate_completed_run "${SMOKE_NAME}" "can" "${ROTATION8_TRANSFORMS}"
  echo "[done] ${SMOKE_NAME} $(date -Iseconds)"
fi

run_variant \
  "flowtte_hplus_dvt_identity_cal4096_all8_${RUN_SUFFIX}" \
  "identity"

run_variant \
  "flowtte_hplus_dvt_superad_rotation8_all8_${RUN_SUFFIX}" \
  "${ROTATION8_TRANSFORMS}"

printf 'run_suffix=%s\nrotation8_smoke=%s\nidentity_control=%s\nrotation8_candidate=%s\nobjects=%s\nsupport_selection=%s\nsupport_manifest_sha256=%s\nmethod_bundle_sha256=%s\nidentity_transforms=identity\nidentity_effective_support_views=16\nrotation8_transforms=%s\nrotation8_effective_support_views=128\ncalibration_sample_size=4096\ncleanup_anomaly_maps=true\n' \
  "${RUN_SUFFIX}" \
  "${SMOKE_NAME}" \
  "flowtte_hplus_dvt_identity_cal4096_all8_${RUN_SUFFIX}" \
  "flowtte_hplus_dvt_superad_rotation8_all8_${RUN_SUFFIX}" \
  "${ALL_OBJECTS}" \
  "${SUPPORT_SELECTION_POLICY}" \
  "${SUPPORT_MANIFEST_SHA256}" \
  "${METHOD_BUNDLE_SHA256}" \
  "${ROTATION8_TRANSFORMS}" \
  >"${GROUP_ROOT}/remote_run_complete.txt"

echo "[complete] H+ DVT rotation-8 ablation ${RUN_SUFFIX} $(date -Iseconds)"
