"""Portable eager-reference coverage for CoRD's recurrent components."""

import importlib.util

import pytest

torch = pytest.importorskip("torch")

from cord import CordConfig
from cord.modules import CordKDA, CordLatentMoE, CordMLA


@pytest.fixture()
def component_config():
    return CordConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_decoder_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        concept_slots=2,
        concept_num_attention_heads=4,
        concept_latent_size=8,
        num_recurrent_loops=1,
        minimum_recurrent_loops=1,
        max_recurrent_loops=1,
        num_experts=2,
        num_experts_per_token=1,
        routed_latent_width=8,
        expert_intermediate_size=16,
        latent_kv_size=8,
        max_position_embeddings=16,
        kda_backend="eager",
    )


def test_eager_kda_chunked_recurrence_matches_full_sequence(component_config):
    torch.manual_seed(3)
    kda = CordKDA(component_config).eval()
    states = torch.randn(2, 5, component_config.hidden_size)
    full_output, full_state = kda(states)
    first_output, first_state = kda(states[:, :2])
    second_output, second_state = kda(states[:, 2:], first_state)
    assert torch.allclose(torch.cat((first_output, second_output), dim=1), full_output, atol=1e-6, rtol=1e-5)
    assert torch.allclose(second_state, full_state, atol=1e-6, rtol=1e-5)


def test_mla_returns_finite_attention_weights(component_config):
    mla = CordMLA(component_config).eval()
    output, attention = mla(torch.randn(2, 4, component_config.hidden_size), output_attentions=True)
    assert output.shape == (2, 4, component_config.hidden_size)
    assert attention is not None and torch.isfinite(attention).all()
    assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)), atol=1e-6)


def test_moe_routes_top_k_and_updates_loss_free_bias(component_config):
    moe = CordLatentMoE(component_config).eval()
    states = torch.randn(2, 3, component_config.hidden_size)
    output, router_logits, counts = moe(states)
    assert output.shape == states.shape
    assert router_logits.shape == (2, 3, component_config.num_experts)
    assert counts.sum().item() == states.shape[0] * states.shape[1] * component_config.num_experts_per_token
    before = moe.e_score_correction_bias.clone()
    moe.update_correction_bias(counts, int(counts.sum().item()))
    assert not torch.equal(before, moe.e_score_correction_bias)


@pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="optional fla-core dependency is unavailable")
def test_fla_kda_matches_eager_reference(component_config):
    fla_config = CordConfig.from_dict({**component_config.to_dict(), "kda_backend": "fla"})
    eager = CordKDA(component_config).eval()
    fla = CordKDA(fla_config).eval()
    fla.load_state_dict(eager.state_dict())
    states = torch.randn(1, 4, component_config.hidden_size)
    eager_output, _ = eager(states)
    fla_output, _ = fla(states)
    assert torch.allclose(fla_output, eager_output, atol=1e-4, rtol=1e-4)
