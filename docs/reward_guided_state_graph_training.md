# Reward-guided recurrent state-graph training

This implementation follows `.reference/reward_guided_recurrent_state_graph.md`
and keeps the ARC protocol boundary explicit.

## Learned graph action

`CordStateController` factorizes one decision as:

```text
p(parent | current, archive, goal, budget)
* p(operator | parent, context)
* p(parent2 | parent, context)       # merge only
```

The five operators are `CONTINUE`, `ROLLBACK`, `BRANCH`, `MERGE`, and `HALT`.
Rollback preserves useful residual information from the current state while
resuming from the selected archived state. Merge uses gated concept-packet
cross-attention, not vector averaging. The controller also predicts state value
and positive uncertainty for every possible parent.

Search is bounded by expansions, beam width, verified leaves, hot archive
capacity, and checkpoint slots. Candidate novelty is cosine distance from the
materialized archive. Fingerprint collisions become cycle-penalized negative
transitions instead of silently disappearing.

## Reward and credit

ARC exact grid reward is `1.0`. A valid wrong grid receives a small validity
reward; cell accuracy is scaled to a bounded shaping term. Expansion, decode,
and cycle costs are separate. After all policy-ranked terminal leaves have been
decoded and externally verified, `backup_graph_returns` propagates discounted
returns through every parent edge in the latent DAG. It then records
`advantage = return - value` on every transition.

The verifier runs **after** terminal selection. It cannot replace the selected
leaf. Evaluation reports both:

- `selected_solution_accuracy`: deployable controller choice;
- `oracle_success_at_n`: whether any externally scored leaf was exact.

It also reports average expansions, decoded leaves/tokens, resource cost, and
latency. Validation/test targets are never inputs to parent, operator, merge,
halt, or leaf-selection decisions.

## Training stages

1. Train the prefix-causal fixed-chain model with SFT.
2. Collect graph trajectories only on the training split.
3. Store detached replay transitions, with successful outcomes and large
   positive/negative advantages sampled more often.
4. Train controller and critic using Monte Carlo targets and AWR. Passing
   `ppo_clip` to `compute_graph_policy_loss` enables clipped PPO after the
   controller is calibrated.
5. Replay at most `config.local_bptt_loops` connected transitions and distill
   their stored child packets through the recurrent core. Positive-return
   terminal paths are recomputed and teacher-forced against the target so token
   CE reaches the recurrent core through the same bounded path. The fixed-chain
   SFT loss remains an anchor. Backbone learning rate defaults to one tenth of
   controller LR.

Run phase 1 only with the normal command. Enable phase 2 with, for example:

```bash
python scripts/train_arc.py \
  --epochs 10 \
  --graph-training-epochs 2 \
  --ppo-training-epochs 1 \
  --graph-max-expansions 4 \
  --graph-beam-size 4 \
  --graph-verified-leaves 4
```

PPO is refused unless at least one AWR epoch runs first. PPO consumes the fresh
on-policy transitions from each batch (not old replay), uses the stored behavior
log-probability for clipping, and applies KL regularization to the frozen best
SFT checkpoint. AWR continues to use the bounded off-policy replay buffer.

The best graph checkpoint contains `graph_replay.pt` alongside model and
trainer state. Search remains no-grad during rollout; gradients are created by
recomputing policy/value predictions and bounded local transitions from replay.
This prevents an unbounded autograd graph while still training the controller,
critic, operator codes, rollback/merge modules, and shared recurrent core.
