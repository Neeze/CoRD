from types import SimpleNamespace
import copy

import pytest

torch = pytest.importorskip("torch")

from dataset.data import EOS_TOKEN, GRID_OFFSET, IGNORE_INDEX, PAD_TOKEN, ROW_SEP
from trainer.trainer import (
    ARCMetrics,
    _generation_inputs,
    decode_arc_grid,
    evaluate,
    is_better_validation,
    train_epoch,
    train_graph_epoch,
    validate_metrics,
)


def _batch_for_generation():
    return {
        "input_ids": torch.tensor(
            [[1, 4, 13, 14, 5, 13, 2, 0], [1, 6, 13, 14, 7, 13, 2, 0]]
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0]]
        ),
        "prefix_lengths": torch.tensor([4, 4]),
        "target_ids": torch.tensor([[5, 13, 2], [7, 13, 2]]),
    }


def test_generation_inputs_exclude_targets_and_right_pad_prefixes():
    batch = _batch_for_generation()
    batch["prefix_lengths"] = torch.tensor([3, 4])
    generation_inputs, max_new_tokens = _generation_inputs(batch)
    assert generation_inputs["input_ids"].tolist() == [[1, 4, 13, PAD_TOKEN], [1, 6, 13, 14]]
    assert generation_inputs["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert generation_inputs["prefix_lengths"].tolist() == [3, 4]
    assert max_new_tokens == 3


def test_metrics_keep_teacher_forced_and_generated_denominators_separate():
    metrics = ARCMetrics()
    tokens = [GRID_OFFSET + 2, ROW_SEP, EOS_TOKEN]
    logits = torch.zeros(1, len(tokens), 16)
    logits[0, torch.arange(len(tokens)), torch.tensor(tokens)] = 5.0
    metrics.update_teacher_forced(logits, torch.tensor([tokens]))
    assert metrics.as_dict()["generated_examples"] == 0.0
    metrics.update_generated("task", tokens, [[2]])
    values = metrics.as_dict()
    assert values["teacher_forced_examples"] == 1.0
    assert values["generated_examples"] == 1.0
    assert values["grid_exact"] == 1.0


def test_validation_metric_order_prefers_task_exact_then_grid_exact():
    best = {"task_exact": 0.0, "grid_exact": 1.0, "loss": 0.1}
    current = {"task_exact": 1.0, "grid_exact": 0.0, "loss": 0.2}
    assert is_better_validation(current, best)


def test_evaluate_uses_prompt_only_generation():
    class Model:
        def __init__(self):
            self.generation_inputs = []

        def eval(self):
            return self

        def __call__(self, *, input_ids, labels, **kwargs):
            logits = torch.zeros(input_ids.shape[0], 3, 16)
            logits[:, 0, 5] = 1.0
            logits[:, 1, ROW_SEP] = 1.0
            logits[:, 2, EOS_TOKEN] = 1.0
            return SimpleNamespace(
                loss=torch.tensor(1.0), logits=logits, active_loops=None,
                halting_probs=None, loop_values=None, loop_uncertainties=None, router_counts=(),
            )

        def generate(self, *, input_ids, **kwargs):
            self.generation_inputs.append(input_ids.clone())
            completion = torch.tensor([[5, ROW_SEP, EOS_TOKEN]]).expand(input_ids.shape[0], -1)
            return torch.cat((input_ids, completion), dim=1)

    batch = _batch_for_generation()
    batch["prefix_lengths"] = torch.tensor([3, 4])
    batch.update({
        "labels": torch.tensor([[-100, -100, -100, -100, 5, ROW_SEP, EOS_TOKEN, -100],
                                [-100, -100, -100, -100, 7, ROW_SEP, EOS_TOKEN, -100]]),
        "task_ids": ["task-a", "task-b"],
        "target_grids": [[[2]], [[2]]],
        "query_indices": [0, 0],
    })
    model = Model()
    metrics = evaluate(model, [batch], torch.device("cpu"), namespace="unit")
    assert [prompt.tolist() for prompt in model.generation_inputs] == [[[1, 4, 13, PAD_TOKEN], [1, 6, 13, 14]]]
    assert metrics["generated_examples"] == 2.0
    assert metrics["grid_exact"] == 1.0


def test_decode_arc_grid_and_metrics():
    tokens = [GRID_OFFSET + 2, GRID_OFFSET + 3, ROW_SEP, EOS_TOKEN]
    assert decode_arc_grid(tokens) == [[2, 3]]
    assert decode_arc_grid([GRID_OFFSET + 1, EOS_TOKEN]) is None
    metrics = ARCMetrics()
    target = torch.tensor([tokens])
    logits = torch.full((1, len(tokens), 32), -100.0)
    logits[0, torch.arange(len(tokens)), target[0]] = 100.0
    metrics.update_teacher_forced(logits, target)
    metrics.update_generated("task-one", tokens, [[2, 3]])
    values = metrics.as_dict()
    assert values["token_accuracy"] == 1.0
    assert values["completion_exact"] == 1.0
    assert values["grid_exact"] == 1.0
    assert values["task_exact"] == 1.0


@pytest.mark.parametrize(
    "tokens",
    [
        [GRID_OFFSET, ROW_SEP],  # missing EOS
        [GRID_OFFSET, EOS_TOKEN],  # EOS before closing the row
        [ROW_SEP, EOS_TOKEN],  # empty row
        [GRID_OFFSET, ROW_SEP, GRID_OFFSET, GRID_OFFSET, ROW_SEP, EOS_TOKEN],  # ragged
        [GRID_OFFSET, 14, ROW_SEP, EOS_TOKEN],  # invalid control token
    ],
)
def test_decode_arc_grid_rejects_invalid_grammar(tokens):
    assert decode_arc_grid(tokens) is None


def test_decode_arc_grid_accepts_multiple_rectangular_rows():
    tokens = [GRID_OFFSET, GRID_OFFSET + 1, ROW_SEP, GRID_OFFSET + 2, GRID_OFFSET + 3, ROW_SEP, EOS_TOKEN]
    assert decode_arc_grid(tokens) == [[0, 1], [2, 3]]


def test_malformed_generation_is_counted():
    metrics = ARCMetrics()
    metrics.update_generated("task", [99, EOS_TOKEN], [[0]])
    values = metrics.as_dict()
    assert values["malformed_outputs"] == 1.0
    assert values["generated_examples"] == 1.0
    assert values["valid_grid_rate"] == 0.0


def test_metric_finiteness_guard_rejects_nan_logits():
    metrics = ARCMetrics()
    logits = torch.zeros(1, 2, 16)
    logits[0, 0, 0] = float("nan")
    with pytest.raises(FloatingPointError, match="non-finite logits"):
        metrics.update_teacher_forced(logits, torch.tensor([[3, 2]]))


def test_metric_validation_rejects_missing_generation():
    metrics = ARCMetrics()
    metrics.update_teacher_forced(torch.zeros(1, 1, 16), torch.tensor([[3]]))
    with pytest.raises(ValueError, match="no generated examples"):
        metrics.validate()


def test_task_exact_requires_every_generated_query_to_match():
    metrics = ARCMetrics()
    valid = [GRID_OFFSET, ROW_SEP, EOS_TOKEN]
    metrics.set_expected_query_counts(["two-query-task", "two-query-task"])
    metrics.update_generated("two-query-task", valid, [[0]])
    metrics.update_generated("two-query-task", valid, [[1]])
    assert metrics.as_dict()["task_exact"] == 0.0


def test_evaluate_rejects_nonfinite_loss_with_batch_context():
    class NonFiniteModel:
        def eval(self):
            return self

        def __call__(self, *, input_ids, **kwargs):
            return SimpleNamespace(loss=torch.tensor(float("nan")), logits=torch.zeros(input_ids.shape[0], 3, 16))

    batch = _batch_for_generation()
    batch.update({
        "labels": torch.full((2, 8), IGNORE_INDEX),
        "task_ids": ["task-a", "task-b"],
        "target_grids": [[[2]], [[2]]],
        "query_indices": [0, 0],
    })
    with pytest.raises(FloatingPointError, match="task_ids=task-a,task-b"):
        evaluate(NonFiniteModel(), [batch], torch.device("cpu"), namespace="unit")


def test_evaluate_rejects_nonfinite_logits_with_maximum_context():
    class NonFiniteModel:
        def eval(self):
            return self

        def __call__(self, *, input_ids, **kwargs):
            logits = torch.zeros(input_ids.shape[0], 3, 16)
            logits[0, 0, 0] = float("inf")
            return SimpleNamespace(loss=torch.tensor(1.0), logits=logits)

    batch = _batch_for_generation()
    batch.update({
        "labels": torch.full((2, 8), IGNORE_INDEX),
        "task_ids": ["task-a", "task-b"],
        "target_grids": [[[2]], [[2]]],
        "query_indices": [0, 0],
    })
    with pytest.raises(FloatingPointError, match="max_abs_logit=inf"):
        evaluate(NonFiniteModel(), [batch], torch.device("cpu"), namespace="unit")


def test_validate_metrics_rejects_incomplete_protocol_state():
    with pytest.raises(ValueError, match="incomplete evaluation metrics"):
        validate_metrics({"loss": 1.0})


def _training_batch(values: list[float]) -> dict[str, object]:
    input_ids = torch.tensor([[int(value)] for value in values])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.full_like(input_ids, 3),
        "prefix_lengths": torch.zeros(len(values), dtype=torch.long),
        "target_ids": torch.full((len(values), 1), 3, dtype=torch.long),
    }


def test_gradient_accumulation_matches_logical_batches_including_tail():
    class ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, *, input_ids, **kwargs):
            values = input_ids.float().squeeze(-1)
            loss = ((self.weight * values - 1.0) ** 2).mean()
            logits = torch.zeros(input_ids.shape[0], 1, 16)
            logits[:, 0, 3] = 1.0
            return SimpleNamespace(
                loss=loss, logits=logits, active_loops=None, halting_probs=None,
                loop_values=None, loop_uncertainties=None, router_counts=(),
            )

    accumulated = ScalarModel()
    logical = ScalarModel()
    logical.load_state_dict(accumulated.state_dict())
    accumulated_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    logical_optimizer = torch.optim.SGD(logical.parameters(), lr=0.1)
    train_epoch(
        accumulated, [_training_batch([1.0]), _training_batch([2.0]), _training_batch([3.0])],
        accumulated_optimizer, torch.device("cpu"), gradient_accumulation_steps=2,
    )
    train_epoch(
        logical, [_training_batch([1.0, 2.0]), _training_batch([3.0])], logical_optimizer,
        torch.device("cpu"), gradient_accumulation_steps=1,
    )
    assert torch.allclose(accumulated.weight, logical.weight)


# Keep torch imported for downstream evaluator stubs without constructing a full model here.
assert SimpleNamespace and IGNORE_INDEX == -100 and PAD_TOKEN == 0


def test_graph_training_epoch_consumes_verified_replay_and_updates_policy():
    from cord import (
        CordConfig, CordForCausalLM, CordGraphReplayBuffer, CordSearchConfig,
        build_graph_optimizer_param_groups,
    )

    torch.manual_seed(3)
    config = CordConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_decoder_layers=1, num_attention_heads=4, num_key_value_heads=2,
        concept_slots=2, concept_num_attention_heads=4, concept_latent_size=8,
        num_recurrent_loops=1, minimum_recurrent_loops=1, max_recurrent_loops=1,
        num_kda_layers=1, num_mla_layers=0, num_experts=1, num_experts_per_token=1,
        routed_latent_width=8, expert_intermediate_size=16, max_position_embeddings=16,
    )
    model = CordForCausalLM(config)
    prompt = torch.tensor([[1, GRID_OFFSET, ROW_SEP, 14]])
    target = torch.tensor([[GRID_OFFSET + 2, ROW_SEP, EOS_TOKEN]])
    input_ids = torch.cat((prompt, target), dim=1)
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": torch.cat((torch.full_like(prompt, IGNORE_INDEX), target), dim=1),
        "prefix_lengths": torch.tensor([prompt.shape[1]]),
        "target_ids": target,
        "target_grids": [[[2]]],
        "task_ids": ["graph-unit"],
        "query_indices": [0],
    }
    groups = build_graph_optimizer_param_groups(model, 1e-3, backbone_lr_scale=0.1)
    optimizer = torch.optim.AdamW(groups)
    replay = CordGraphReplayBuffer(32)
    before = model.model.state_controller.parent_selector.weight.detach().clone()
    step, metrics, replay = train_graph_epoch(
        model,
        [batch],
        optimizer,
        torch.device("cpu"),
        search_config=CordSearchConfig(max_expansions=1, beam_size=2, max_verified_leaves=2),
        replay_buffer=replay,
        replay_batch_size=2,
        decode_max_new_tokens=8,
    )
    after = model.model.state_controller.parent_selector.weight.detach()
    assert step == 1 and len(replay) > 0
    assert metrics["graph_rollouts"] == 1.0
    assert metrics["replay_size"] > 0.0
    assert not torch.equal(before, after)

    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    step, ppo_metrics, replay = train_graph_epoch(
        model,
        [batch],
        optimizer,
        torch.device("cpu"),
        search_config=CordSearchConfig(max_expansions=1, beam_size=2, max_verified_leaves=2),
        replay_buffer=replay,
        replay_batch_size=2,
        global_step=step,
        decode_max_new_tokens=8,
        ppo_clip=0.2,
        reference_model=reference,
        kl_weight=0.01,
    )
    assert step == 2
    assert ppo_metrics["graph_policy_algorithm"] == 1.0
    assert ppo_metrics["graph_reference_kl"] >= 0.0
