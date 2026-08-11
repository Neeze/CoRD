# ARC-AGI-2 supervised training audit

## Delivered protocol

`python scripts/train_arc.py` trains the registered `CordForCausalLM` prototype on the public ARC-AGI-2 corpus using the prefix-causal path implemented in `src/cord/modeling_cord.py`.

- Only `datasets/ARC-AGI-2/training/*.json` is split at the task-file level. The deterministic default split is 80:20 (800 train and 200 validation tasks for the local corpus).
- `datasets/ARC-AGI-2/evaluation/*.json` is reserved for final test evaluation. It is not used to select checkpoints.
- A sample is one query per task. Demonstrations plus the query input are the concept-encoder prefix; only query output tokens and EOS have labels.
- Train and validation use task-consistent color-permutation and dihedral augmentation. Training augmentation changes deterministically with epoch; validation augmentation is fixed and repeatable. Test augmentation is disabled.
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

TensorBoard includes teacher-forced token-weighted loss/perplexity/accuracy, autoregressive generated-grid validity/exactness, model diagnostics, router usage, optimizer state, and system throughput. Validation and final test generation use one right-padded prompt-only batch, deterministic greedy decoding, the ARC EOS token, and the largest target-completion cap in that batch. Teacher-forced completion exact match is diagnostic; generated grid/task exactness is the ARC quality metric used for checkpoint selection.

The evaluator fails fast on non-finite losses, logits, or aggregate metrics. A run with no supervised targets or no generated queries is a protocol failure rather than a valid zero/NaN result. Final artifacts use `allow_nan=False` and record both metric families plus the checkpoint selection rule.

Before a full run, use a tiny one-task overfit/generation test. A decreasing teacher-forced loss alone is not evidence that the model can generate a valid ARC grid.

The current CLI still uses unconstrained greedy generation. Malformed, missing-EOS, or invalid-token completions are counted as failures. Grammar-constrained decoding is a separate future ablation, not part of the baseline score.

The state-graph search remains a separate no-grad research path. It can now decode a bounded set of terminal concept leaves through the normal decoder cache and score them with the typed offline ARC verifier. Each returned trajectory includes parent IDs, operator, decoded tokens, verifier outcome, seed, and resource cost. This is a measurement seam only: it does not train or tune the model against evaluation tasks.

`scripts/evaluate_arc_search.py --checkpoint <best-checkpoint>` reports direct greedy and graph-search outcomes on the deterministic training-validation partition only. Its report deliberately includes decoded-leaf count and resource cost so a multi-leaf search is not presented as compute-equivalent to one direct decode. Phase 5 remains gated until this report demonstrates a reproducible, meaningful held-out comparison.

The distinction is deliberate: teacher forcing measures token prediction under the correct target prefix; autoregressive generation measures actual ARC solving without target leakage.


## Audit boundary and research gaps

This is a supervised prefix-causal ARC data and training pipeline, not an implementation of the proposed reward-guided recurrent state graph research program. Current CoRD state search is no-grad heuristic inference. The repository does **not** yet implement:

- learned state/operator controller or transition Q policy;
- return backup, TD/Monte-Carlo state targets, advantage routing, or search trajectory replay/distillation;
- graph-local BPTT with replayable archived transitions;
- truly bounded cold checkpoint storage across unlimited archive depth;
- calibrated value/uncertainty/halting losses from ARC targets;
- Muon, distributed training/expert-parallel reduction, or production fused attention/KDA kernels.

The value, uncertainty, halting, and router diagnostics logged by the pipeline therefore must not be interpreted as validated state-graph quality metrics. These require separately designed targets, verifier contracts, baselines, and evaluation.
