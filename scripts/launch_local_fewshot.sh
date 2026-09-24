#!/usr/bin/env bash
set -euo pipefail

# Local RTX 4090 Few-shot Scaling Runner
# Covers Classic MVTec AD 1 and ViSA few-shot experiments (1, 2, 4, 8 shots)

PYTHON=".venv/bin/python"
MVTEC_DATA="/home/hun/Volume/DATA/MVTecAD"
VISA_DATA="/home/hun/Volume/DATA/VisA_pytorch/1cls"

OUT_ROOT="results/benchmark_runs/fewshot"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "Starting Local Few-shot Scaling on RTX 4090 (cuda:0)"
echo "Target: Classic MVTec AD 1 (15 classes) & ViSA (12 classes)"
echo "Time: $(date)"
echo "============================================================"

# --- Phase 1: MVTec AD 1 (15 classes) ---
MVTEC_RUNS=(
    "mvtec_dinov2_2shot"
    "mvtec_dinov2_4shot"
    "mvtec_dinov2_8shot"
    "mvtec_dinov2_1shot"
    "mvtec_dinov3_2shot"
    "mvtec_dinov3_4shot"
    "mvtec_dinov3_8shot"
    "mvtec_dinov3_1shot"
)

for run_name in "${MVTEC_RUNS[@]}"; do
    cfg="configs/benchmarks/fewshot/${run_name}.yaml"
    out="${OUT_ROOT}/mvtec/${run_name}"
    log="${LOG_DIR}/${run_name}.log"
    echo ""
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Launching ${run_name}..."
    ${PYTHON} run.py \
        --config "${cfg}" \
        --data-root "${MVTEC_DATA}" \
        --output-dir "${out}" \
        --device cuda:0 \
        > "${log}" 2>&1
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Completed ${run_name}."
done

# --- Phase 2: ViSA (12 classes) ---
VISA_RUNS=(
    "visa_dinov2_2shot"
    "visa_dinov2_4shot"
    "visa_dinov2_8shot"
    "visa_dinov2_1shot"
    "visa_dinov3_2shot"
    "visa_dinov3_4shot"
    "visa_dinov3_8shot"
    "visa_dinov3_1shot"
)

for run_name in "${VISA_RUNS[@]}"; do
    cfg="configs/benchmarks/fewshot/${run_name}.yaml"
    out="${OUT_ROOT}/visa/${run_name}"
    log="${LOG_DIR}/${run_name}.log"
    echo ""
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Launching ${run_name}..."
    ${PYTHON} run.py \
        --config "${cfg}" \
        --data-root "${VISA_DATA}" \
        --output-dir "${out}" \
        --device cuda:0 \
        > "${log}" 2>&1
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Completed ${run_name}."
done

echo ""
echo "============================================================"
echo "All local few-shot benchmarks completed successfully!"
echo "Time: $(date)"
echo "============================================================"
