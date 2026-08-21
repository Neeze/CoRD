# ARC-AGI-2 supervised training audit

## Delivered protocol

`python scripts/train_arc.py` trains the registered `CordForCausalLM` prototype on the public ARC-AGI-2 corpus using the prefix-causal path implemented in `src/cord/modeling_cord.py`.

- Only `datasets/ARC-AGI-2/training/*.json` is split at the task-file level. The deterministic default split is 80:20 (800 train and 200 validation tasks for the local corpus).
- `datasets/ARC-AGI-2/evaluation/*.json` is reserved for final test evaluation. It is not used to select checkpoints.
- A sample is one query per task. Demonstrations plus the query input are the concept-encoder prefix; only query output tokens and EOS have labels.
- Train keeps each canonical task/query and adds `--num-aug` task-consistent color-permutation plus dihedral variants per epoch. Their descriptors change deterministically with epoch; validation and test are canonical/unaugmented.
- The loader validates ARC grids and refuses silent truncation. Use `--max-length` only as a deliberate safety gate.
- The checked-in configuration `configs/cord-50m.json` is registered through `cord` and must contain strictly fewer than 50,000,000 unique parameters. Tied embeddings are counted once.

## Run and monitor

```bash
.venv/bin/python scripts/train_arc.py \
  --data-dir datasets/ARC-AGI-2 \
  --epochs 10 \
  --batch-size 2

tensorboard --logdir runs --host 127.0.0.1
```

Use `--smoke-optimizer-steps N` only for a deliberately limited smoke run. Each run writes split metadata, parameter count, length limits, checkpoint, and final test metrics to its output directory.

`--num-aug N` means N additional train copies per query (plus one canonical copy), so it multiplies samples and optimizer work by `N + 1`. Start with `--num-aug 1`; use 4 only after confirming throughput and memory on the target GPU. The D4/color transforms are task-consistent. Unlike HRM's fixed 30×30 canvas, CoRD does not apply translational padding because its ragged-grid serialization has no safe inverse translation.

TensorBoard includes teacher-forced token-weighted loss/perplexity/accuracy, autoregressive generated-grid validity/exactness, model diagnostics, router usage, optimizer state, and system throughput. Validation and final test generation use one right-padded prompt-only batch, deterministic greedy decoding, the ARC EOS token, and the largest target-completion cap in that batch. Teacher-forced completion exact match is diagnostic; generated grid/task exactness is the ARC quality metric used for checkpoint selection.

The evaluator fails fast on non-finite losses, logits, or aggregate metrics. A run with no supervised targets or no generated queries is a protocol failure rather than a valid zero/NaN result. Final artifacts use `allow_nan=False` and record both metric families plus the checkpoint selection rule.

Before a full run, use a tiny one-task overfit/generation test. A decreasing teacher-forced loss alone is not evidence that the model can generate a valid ARC grid.

The current CLI still uses unconstrained greedy generation. Malformed, missing-EOS, or invalid-token completions are counted as failures. Grammar-constrained decoding is a separate future ablation, not part of the baseline score.

The state-graph rollout remains no-grad so archive depth does not create an unbounded autograd graph. It now uses a learned hierarchical parent/operator/merge controller, decodes a bounded set of policy-ranked terminal leaves, applies the typed offline ARC verifier only after selection, backs terminal returns through the DAG, and emits replayable transitions. `--graph-training-epochs` recomputes controller/value losses and bounded local transitions from training-split replay.

`scripts/evaluate_arc_search.py --checkpoint <best-checkpoint>` reports direct greedy and graph-search outcomes on the deterministic training-validation partition only. Its report deliberately includes decoded-leaf count and resource cost so a multi-leaf search is not presented as compute-equivalent to one direct decode. Phase 5 remains gated until this report demonstrates a reproducible, meaningful held-out comparison.

The distinction is deliberate: teacher forcing measures token prediction under the correct target prefix; autoregressive generation measures actual ARC solving without target leakage.


## Remaining scale-up gaps

The 50M pipeline implements phase-one search distillation/AWR, Monte Carlo value targets, uncertainty calibration, learned halt actions, replay, and local truncated BPTT. Clipped PPO is available as a loss mode but is deliberately not the default before value calibration. Native DDP supports SFT and graph phases with distributed sampling, gradient/router-stat synchronization, and rank-zero evaluation/checkpointing. Remaining scale-up work includes truly bounded cold storage across unlimited archive depth, Muon, FSDP/DeepSpeed and expert-parallel sharding, production fused attention/KDA kernels, and empirical calibration at sufficient rollout volume.

Value, uncertainty, halt, and policy diagnostics become meaningful only after graph training; randomly initialized heads from an SFT-only checkpoint are still uncalibrated.
