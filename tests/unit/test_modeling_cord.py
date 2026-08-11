import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cord import CordConfig, CordForCausalLM, CordModel, CordSearchConfig


@pytest.fixture()
def tiny_config():
    return CordConfig(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_decoder_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        concept_slots=6,
        concept_num_attention_heads=4,
        concept_latent_size=16,
        num_recurrent_loops=2,
        minimum_recurrent_loops=2,
        max_recurrent_loops=4,
        num_experts=4,
        num_experts_per_token=2,
        routed_latent_width=16,
        expert_intermediate_size=32,
        max_position_embeddings=32,
        use_cache=True,
    )


def test_config_round_trip(tiny_config):
    restored = CordConfig.from_dict(tiny_config.to_dict())
    assert restored.to_dict() == tiny_config.to_dict()


def test_prototype_config_is_packaged():
    path = Path(__file__).parents[2] / "configs" / "cord-50m.json"
    config = CordConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert config.concept_slots == 64
    assert config.num_experts == 8


def test_model_outputs_and_backward(tiny_config):
    model = CordModel(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 7))
    outputs = model(
        input_ids,
        prefix_lengths=torch.tensor([3, 3]),
        output_hidden_states=True,
        return_dict=True,
    )
    assert outputs.last_hidden_state.shape == (2, 4, 32)
    assert outputs.concept_states.shape == (2, 6, 32)
    assert outputs.loop_values.shape == (2, 2)
    outputs.last_hidden_state.square().mean().backward()


def test_attention_mask_excludes_padded_completion_tokens(tiny_config):
    model = CordModel(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 6))
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0]])
    outputs = model(
        input_ids,
        attention_mask=attention_mask,
        prefix_lengths=torch.tensor([3, 3]),
        use_cache=False,
    )
    changed_padding = input_ids.clone()
    changed_padding[:, 4:] = torch.randint(0, tiny_config.vocab_size, (2, 2))
    changed_outputs = model(
        changed_padding,
        attention_mask=attention_mask,
        prefix_lengths=torch.tensor([3, 3]),
        use_cache=False,
    )
    assert outputs.last_hidden_state.shape == (2, 1, tiny_config.hidden_size)
    assert torch.allclose(outputs.last_hidden_state, changed_outputs.last_hidden_state)


def test_prefix_causal_lm_loss_save_load_and_tied_embeddings(tmp_path, tiny_config):
    model = CordForCausalLM(tiny_config).eval()
    input_ids = torch.randint(0, tiny_config.vocab_size, (2, 7))
    labels = input_ids.clone()
    labels[:, :3] = -100
    outputs = model(input_ids, prefix_lengths=torch.tensor([3, 3]), labels=labels)
    assert outputs.loss.ndim == 0
    assert torch.isfinite(outputs.loss)
    assert outputs.logits.shape[:2] == (2, 4)
    assert outputs.loss_breakdown["token"].ndim == 0
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    model.save_pretrained(tmp_path)
    restored = CordForCausalLM.from_pretrained(tmp_path).eval()
    restored_outputs = restored(input_ids, prefix_lengths=torch.tensor([3, 3]), labels=labels)
    assert torch.allclose(outputs.logits, restored_outputs.logits)


def test_tuple_output_and_auto_classes(tiny_config):
    model = CordForCausalLM(tiny_config)
    input_ids = torch.randint(0, tiny_config.vocab_size, (1, 5))
    outputs = model(input_ids, prefix_lengths=torch.tensor([2]), return_dict=False)
    assert isinstance(outputs, tuple)
    assert outputs[0].shape[-1] == tiny_config.vocab_size
    assert transformers.AutoConfig.for_model("cord").model_type == "cord"
    assert isinstance(transformers.AutoModel.from_config(tiny_config), CordModel)
    assert isinstance(transformers.AutoModelForCausalLM.from_config(tiny_config), CordForCausalLM)


def test_generate_accepts_transformers_empty_cache(tiny_config):
    model = CordForCausalLM(tiny_config).eval()
    prompt = torch.randint(0, tiny_config.vocab_size, (2, 4))
    prompt[0, -1] = tiny_config.pad_token_id
    generated = model.generate(
        input_ids=prompt,
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        prefix_lengths=torch.tensor([3, 4]),
        max_new_tokens=2,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=2,
        pad_token_id=0,
    )
    assert generated.shape[0] == 2
    assert generated.shape[1] >= prompt.shape[1]


def test_cached_decode_matches_full_decode(tiny_config):
    model = CordForCausalLM(tiny_config).eval()
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 3))
    first = model(prompt, prefix_lengths=torch.tensor([3]), use_cache=True)
    bos = torch.full((1, 1), tiny_config.bos_token_id, dtype=torch.long)
    next_token = torch.randint(0, tiny_config.vocab_size, (1, 1))
    cached = model(
        next_token,
        prefix_lengths=torch.tensor([3]),
        past_key_values=first.past_key_values,
        use_cache=True,
    )
    full = model(
        torch.cat((prompt, bos, next_token), dim=1),
        prefix_lengths=torch.tensor([3]),
        use_cache=False,
    )
    assert torch.allclose(cached.logits[:, -1], full.logits[:, -1], atol=1e-5, rtol=1e-4)


def test_state_graph_search(tiny_config):
    model = CordModel(tiny_config).eval()
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
    result = model.search(
        prompt,
        prefix_lengths=torch.tensor([4]),
        search_config=CordSearchConfig(max_expansions=2, beam_size=2),
    )
    assert result.best_state.ndim == 2
    assert result.num_expansions is not None


def test_causal_lm_search_decodes_leaf_with_its_cache(tiny_config):
    model = CordForCausalLM(tiny_config).eval()
    prompt = torch.randint(0, tiny_config.vocab_size, (1, 4))
    result = model.search(
        prompt,
        prefix_lengths=torch.tensor([4]),
        search_config=CordSearchConfig(max_expansions=1, beam_size=1),
        max_new_tokens=2,
    )
    assert result.decoded_tokens is not None
    assert len(result.decoded_tokens) == 1
    assert 1 <= len(result.decoded_tokens[0]) <= 2


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="divisible"):
        CordConfig(hidden_size=30, num_attention_heads=4)
