import pytest

torch = pytest.importorskip("torch")

from cord.controller import CordStateController
from cord.state_graph import CordOperator, CordStateRecord, backup_graph_returns, state_fingerprint
from cord.training import (
    CordGraphReplayBuffer,
    CordGraphTransition,
    compute_graph_policy_loss,
    compute_local_transition_loss,
    build_graph_optimizer_param_groups,
)


def _transition(hidden_size=8):
    current = torch.randn(2, hidden_size)
    archive = torch.randn(3, 2, hidden_size)
    return CordGraphTransition(
        current_state=current,
        archive_states=archive,
        goal_state=torch.randn(2, hidden_size),
        budget_remaining=0.5,
        parent_index=1,
        operator_index=list(CordOperator).index(CordOperator.MERGE),
        second_parent_index=2,
        behavior_log_prob=-2.0,
        return_value=1.0,
        advantage=0.75,
        parent_state=archive[1],
        second_parent_state=archive[2],
        child_state=torch.randn(2, hidden_size),
        exact_reward=1.0,
    )


def test_hierarchical_controller_scores_all_action_factors():
    controller = CordStateController(8, len(CordOperator))
    transition = _transition()
    output = controller(
        transition.current_state,
        transition.archive_states,
        transition.goal_state,
        transition.budget_remaining,
    )
    assert output.parent_logits.shape == (1, 3)
    assert output.operator_logits.shape == (1, 3, len(CordOperator))
    assert output.second_parent_logits.shape == (1, 3, 3)
    assert output.values.shape == output.uncertainties.shape == (1, 3)
    assert torch.isfinite(output.uncertainties).all() and (output.uncertainties > 0).all()
    log_prob = controller.action_log_prob(
        output,
        torch.tensor([1]),
        torch.tensor([transition.operator_index]),
        torch.tensor([2]),
    )
    assert log_prob.shape == (1,) and torch.isfinite(log_prob).all()


def test_awr_value_uncertainty_loss_trains_controller():
    controller = CordStateController(8, len(CordOperator))
    loss, breakdown = compute_graph_policy_loss(controller, [_transition()])
    loss.backward()
    assert torch.isfinite(loss)
    assert set(breakdown) == {
        "graph_policy", "graph_value", "graph_uncertainty", "graph_entropy", "graph_advantage_weight"
    }
    assert controller.parent_selector.weight.grad is not None
    assert controller.operator_policy[-1].weight.grad is not None
    assert controller.second_parent_selector.weight.grad is not None
    assert controller.value_head[-1].weight.grad is not None


def test_clipped_ppo_mode_uses_behavior_log_probability():
    controller = CordStateController(8, len(CordOperator))
    transition = _transition()
    loss, breakdown = compute_graph_policy_loss(
        controller, [transition], ppo_clip=0.2
    )
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(breakdown["graph_policy"])
    assert controller.operator_policy[-1].weight.grad is not None


def test_terminal_return_is_backed_up_without_online_verifier_selection():
    root_state = torch.zeros(2, 4)
    child_state = torch.ones(2, 4)
    records = {
        "root": CordStateRecord(
            "root", (), CordOperator.CONTINUE, 0, 0.1, 0.0, 0.0, 1.0,
            state_fingerprint(root_state), 0, state=root_state,
        ),
        "terminal": CordStateRecord(
            "terminal", ("root",), CordOperator.HALT, 1, 0.2, 1.0, 0.0, 1.0,
            state_fingerprint(child_state), 1, decoded_tokens=(1, 2), state=child_state,
            exact_reward=1.0, terminal=True,
        ),
    }
    backup_graph_returns(
        records, discount=0.9, compute_cost_weight=0.1, decode_cost_weight=0.01
    )
    assert records["terminal"].return_value == pytest.approx(0.98)
    assert records["root"].return_value == pytest.approx(0.782)
    assert records["terminal"].advantage == pytest.approx(0.78)


def test_replay_buffer_and_local_transition_bptt():
    transition = _transition()
    transition.operator_index = list(CordOperator).index(CordOperator.CONTINUE)
    transition.second_parent_state = None
    replay = CordGraphReplayBuffer(capacity=2)
    replay.extend([transition])
    sample = replay.sample(1)
    assert len(sample) == 1 and sample[0].current_state.device.type == "cpu"

    class TransitionCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))

        def _operator_state(self, state, operator, other_state=None):
            return state * self.scale

    core = TransitionCore()
    loss = compute_local_transition_loss(core, sample)
    loss.backward()
    assert loss.item() > 0.0 and core.scale.grad is not None


def test_graph_optimizer_uses_lower_backbone_learning_rate():
    from cord import CordConfig, CordForCausalLM

    config = CordConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_decoder_layers=1, num_attention_heads=4, concept_slots=2,
        concept_num_attention_heads=4, concept_latent_size=8, num_recurrent_loops=1,
        max_recurrent_loops=1, num_kda_layers=1, num_mla_layers=0, num_experts=1,
        num_experts_per_token=1, routed_latent_width=8, expert_intermediate_size=16,
        max_position_embeddings=16,
    )
    model = CordForCausalLM(config)
    groups = build_graph_optimizer_param_groups(
        model, 1e-3, backbone_lr_scale=0.1
    )
    assert {group["component"] for group in groups} == {"controller", "backbone"}
    assert {group["lr"] for group in groups if group["component"] == "controller"} == {1e-3}
    assert {group["lr"] for group in groups if group["component"] == "backbone"} == {1e-4}


def test_replayed_leaf_token_loss_backpropagates_into_concept_state():
    from cord import CordConfig, CordForCausalLM

    config = CordConfig(
        vocab_size=16, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_decoder_layers=1, num_attention_heads=4, concept_slots=2,
        concept_num_attention_heads=4, concept_latent_size=8, num_recurrent_loops=1,
        max_recurrent_loops=1, num_kda_layers=1, num_mla_layers=0, num_experts=1,
        num_experts_per_token=1, routed_latent_width=8, expert_intermediate_size=16,
        max_position_embeddings=16,
    )
    model = CordForCausalLM(config)
    leaf = torch.randn(2, 16, requires_grad=True)
    loss = model.concept_state_loss(leaf, torch.tensor([5, 13, 2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert leaf.grad is not None and leaf.grad.abs().sum() > 0
