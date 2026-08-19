#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-50}"
NUM_AUG="${NUM_AUG:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
NUM_WORKERS="${NUM_WORKERS:-4}"

cd "$ROOT_DIR"
exec "$PYTHON_BIN" scripts/train_arc.py \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --num-aug "$NUM_AUG" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay "$WEIGHT_DECAY" \
  --warmup-ratio "$WARMUP_RATIO" \
  --max-grad-norm "$MAX_GRAD_NORM" \
  --num-workers "$NUM_WORKERS" \
  "$@"
