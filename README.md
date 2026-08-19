# CoRD Transformers

Standalone Hugging Face Transformers project for the Concept-over-Recurrent-Depth (CoRD) language model.

## Quickstart (All-in-One Setup & Training)

To automatically install `uv` (if needed), sync Python dependencies, tune environment optimizations for high-performance hardware (e.g. RTX 5090 GPU / AMD EPYC CPU), and start model training in a single command, run:

```bash
./scripts/bash/setup_and_train.sh
```

### Custom Training Arguments

You can pass extra flags or override environment variables directly:

```bash
DEVICE=cuda BATCH_SIZE=16 ./scripts/bash/setup_and_train.sh --epochs 50 --num-workers 80
```

## Project layout

```text
cord/
├── configs/                 # Reproducible model configurations
├── docs/                    # Architecture and implementation notes
├── scripts/                 # Developer, bash scripts, and training utilities
│   ├── bash/
│   │   ├── setup_and_train.sh   # All-in-one setup & training execution script
│   │   └── train_model.sh       # Underlying training bash wrapper
│   └── train_arc.py             # ARC dataset training script
├── src/cord/                # Installable Python package
│   ├── configuration_cord.py
│   ├── modeling_cord.py
│   ├── modules.py
│   ├── state_graph.py
│   ├── cache_utils.py
│   ├── outputs.py
│   ├── training.py
│   └── __init__.py
├── tests/
│   ├── unit/
│   └── integration/
└── pyproject.toml
```

The first development target is the approximately 49M parameter prototype described in `configs/cord-50m.json`. The 1B configuration from the research blueprint is not used by default in tests.

## Environment Setup via uv

You can also manually sync the virtual environment with [Astral uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The optional `kda` extra enables FLA kernels:

```bash
uv sync --extra kda
```

## Reward-guided graph training

After establishing an SFT checkpoint, enable controller/value training and
local recurrent replay with `--graph-training-epochs`. See
`docs/reward_guided_state_graph_training.md` for the reward, credit-assignment,
evaluation, and target-leakage contracts.
