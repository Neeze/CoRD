#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"

cd "$ROOT_DIR"
exec $PYTHON_BIN scripts/train_arc.py \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  "$@"
