# CoRD architecture status

CoRD is organized as a standalone Transformers package. The model is built in
two public execution paths:

1. `forward()` implements prefix-causal language modeling. Prompt tokens are
   compressed into a fixed concept workspace and completion tokens are decoded
   autoregressively from that workspace.
2. `search()` implements reward-guided state-graph expansion over detached
   concept packets. It owns the hot frontier, cold archive, rollback, branch,
   merge and halt operations. Rollback selectively gates an archived ancestor;
   merge uses gated cross-attention from one concept packet into another rather
   than averaging their states.

The concept codec uses learned slot queries with multi-head attention in the
configured latent width. It attends only to the prompt prefix, preventing
future-token leakage. The decoder uses causal self-attention and concept
cross-attention.

For supervised calls, the first non-ignored label marks the completion start;
the decoder consumes `BOS + target[:-1]` and the loss is aligned to `target`,
which preserves causal training without future-token leakage.

The shared recurrent core follows the research blueprint: three KDA layers and
one gated MLA layer, repeated with shared weights. Latent MoE, SiTU/SwiGLU
selection and bounded Block AttnRes are added behind configuration flags so the
50M prototype can be profiled before scaling.

The optional FLA KDA backend is deliberately not treated as validated unless it
is installed and its parity test runs on compatible hardware. The eager KDA
reference is the portable correctness path.

## Reference implementations

- `resources/kimi_k3/modeling_kimi_linear.py` informs KDA, MLA, cache, MoE and
  Attention Residual interfaces.
- `resources/deepseek_v4/modular_deepseek_v4.py` informs Transformers model
  outputs, router correction bias and stateful generation conventions.
- `resources/large_concept_model/` informs concept-space model organization.
- `resources/cord_1b_2026_training_blueprint_1.md` is the architecture source
  of truth for the CoRD prototype and scale-up targets.
