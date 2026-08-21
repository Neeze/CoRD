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
./scripts/bash/train_model.sh --strategy single --devices 0 --batch-size 8
```

This defaults to 50 epochs, 4 augmentations, gradient accumulation 1,
learning rate `2e-4`, weight decay `0.1`, warmup ratio `0.05`, max gradient
norm `1.0`, and 4 data-loader workers. `--batch-size`, `--strategy`, and
`--devices` are explicit CLI flags; later CLI arguments override wrapper defaults.

For native DDP (the batch size is per GPU):

```bash
./scripts/bash/train_model.sh \
  --strategy ddp \
  --devices 0 1 \
  --batch-size 2 \
  --gradient-accumulation-steps 2
```

This plain invocation relaunches itself through `torchrun`. The example has an
effective batch size of `2 × 2 GPUs × 2 accumulation = 8`. DDP shards the
training dataset, synchronizes model gradients and router statistics, and only
rank zero evaluates, logs, and writes checkpoints. `fsdp` and
`deepspeed_stage_3` are reserved strategy names and currently fail explicitly
because graph replay is not yet compatible with their parameter sharding.

Before optimization starts, the trainer prints a manifest containing input,
prefix, and supervised tokens per epoch; scheduled totals; graph-decode token
budget; optimizer steps; and effective global batch size. Rank zero also keeps
one overall `SFT > AWR > PPO` progress bar with the active segment and actual
processed input-token count, while the transient per-epoch bar remains below it.

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
