#!/usr/bin/env bash
# ==============================================================================
# All-in-One Setup & Training Script for CoRD Model
# - Installs Astral 'uv' if not present
# - Sets up Python virtual environment & dependencies via uv
# - Optimizes system environment for high-performance hardware (RTX 5090 + EPYC)
# - Runs train_model.sh with user specified hyperparameters
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "============================================================"
echo " [1/4] Checking & Installing Astral 'uv' Package Manager"
echo "============================================================"

# Ensure local bin paths are available in PATH
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "[INFO] 'uv' not found in PATH. Installing uv via Astral installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv &> /dev/null; then
    echo "[ERROR] Failed to locate or install 'uv'." >&2
    exit 1
fi

echo "[SUCCESS] uv version: $(uv --version)"

echo ""
echo "============================================================"
echo " [2/4] Setting up Python Environment & Syncing Dependencies"
echo "============================================================"

# Sync dependencies defined in pyproject.toml / uv.lock
uv sync

# Activate environment in current subshell
if [ -f "$ROOT_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.venv/bin/activate"
fi

echo "[SUCCESS] Environment synced successfully."

echo ""
echo "============================================================"
echo " [3/4] Configuring High-Performance Hardware Optimizations"
echo "============================================================"

# Environment variables for PyTorch & Hardware Optimization
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export PYTHONUNBUFFERED=1

# PyTorch CUDA Memory Allocator optimization to prevent fragmentation
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Multi-threading tuning to avoid CPU over-subscription in data-loader workers
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# Use python from synced .venv
export PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

echo "Configuration:"
echo " - GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
echo " - NUM_WORKERS: $NUM_WORKERS"
echo " - PyTorch Allocator: $PYTORCH_CUDA_ALLOC_CONF"
echo " - Python Binary: $PYTHON_BIN"

echo ""
echo "============================================================"
echo " [4/4] Launching Training Script"
echo "============================================================"

chmod +x "$SCRIPT_DIR/train_model.sh"

exec "$SCRIPT_DIR/train_model.sh" "$@"
