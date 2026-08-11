"""Training utility coverage independent of the ARC CLI."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cord import CordConfig, CordForCausalLM
from cord.training import build_cord_optimizer_param_groups, update_router_biases


def _config():
    return CordConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_decoder_layers=1, num_attention_heads=4, num_key_value_heads=2,
        concept_slots=2, concept_num_attention_heads=4, concept_latent_size=8,
        num_recurrent_loops=1, minimum_recurrent_loops=1, max_recurrent_loops=1,
        num_experts=2, num_experts_per_token=1, routed_latent_width=8,
        expert_intermediate_size=16, max_position_embeddings=16,
    )


def test_optimizer_groups_partition_trainable_parameters():
    model = CordForCausalLM(_config())
    groups = build_cord_optimizer_param_groups(model)
    grouped = {id(parameter) for group in groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert grouped == expected
    assert len(grouped) == sum(len(group["params"]) for group in groups)


def test_router_bias_update_accepts_recurrent_router_counts():
    model = CordForCausalLM(_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    outputs = model(input_ids, prefix_lengths=torch.tensor([3]), use_cache=False)
    before = [module.e_score_correction_bias.clone() for module in model.modules() if hasattr(module, "e_score_correction_bias")]
    assert outputs.router_counts
    update_router_biases(model, outputs.router_counts)
    after = [module.e_score_correction_bias for module in model.modules() if hasattr(module, "e_score_correction_bias")]
    assert any(not torch.equal(left, right) for left, right in zip(before, after))
