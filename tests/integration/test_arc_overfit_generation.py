"""End-to-end CPU gate for ARC serialization, generation, and scoring."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cord import CordConfig, CordForCausalLM, CordSearchConfig
from dataset.data import BOS_TOKEN, EOS_TOKEN, GRID_OFFSET, IGNORE_INDEX, IO_SEP, ROW_SEP
from trainer.trainer import decode_arc_grid, evaluate
from trainer.arc_search_evaluation import compare_direct_and_graph


def test_tiny_cord_overfits_one_arc_completion_and_generates_it(tmp_path):
    """A one-example ladder must work before full ARC losses are interpreted."""
    torch.manual_seed(0)
    config = CordConfig(
        vocab_size=16,
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
        num_kda_layers=1,
        num_mla_layers=0,
        num_experts=1,
        num_experts_per_token=1,
        routed_latent_width=8,
        expert_intermediate_size=16,
        max_position_embeddings=16,
        loop_dropout=0.0,
        attention_dropout=0.0,
    )
    model = CordForCausalLM(config)
    prompt = torch.tensor([[BOS_TOKEN, GRID_OFFSET, ROW_SEP, IO_SEP]])
    target = torch.tensor([[GRID_OFFSET + 2, ROW_SEP, EOS_TOKEN]])
    input_ids = torch.cat((prompt, target), dim=1)
    labels = torch.cat((torch.full_like(prompt, IGNORE_INDEX), target), dim=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

    model.train()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids, labels=labels, prefix_lengths=torch.tensor([prompt.shape[1]])).loss
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    model.eval()
    generated = model.generate(
        prompt,
        attention_mask=torch.ones_like(prompt),
        prefix_lengths=torch.tensor([prompt.shape[1]]),
        max_new_tokens=target.shape[1],
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=EOS_TOKEN,
        pad_token_id=0,
    )
    completion = generated[0, prompt.shape[1]:].tolist()
    assert decode_arc_grid(completion) == [[2]]

    evaluation_batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "prefix_lengths": torch.tensor([prompt.shape[1]]),
        "target_ids": target,
        "target_grids": [[[2]]],
        "task_ids": ["00000000"],
        "query_indices": [0],
    }
    before_reload = evaluate(model, [evaluation_batch], torch.device("cpu"), namespace="overfit")
    model.save_pretrained(tmp_path)
    reloaded = CordForCausalLM.from_pretrained(tmp_path).eval()
    after_reload = evaluate(reloaded, [evaluation_batch], torch.device("cpu"), namespace="overfit")
    assert before_reload["grid_exact"] == after_reload["grid_exact"] == 1.0
    assert before_reload["task_exact"] == after_reload["task_exact"] == 1.0

    comparison = compare_direct_and_graph(
        reloaded,
        [evaluation_batch],
        torch.device("cpu"),
        search_config=CordSearchConfig(max_expansions=0, beam_size=1, max_verified_leaves=1),
    )
    assert comparison["examples"] == 1.0
    assert comparison["direct_grid_exact"] == 1.0
    assert comparison["graph_decoded_leaves"] == 1.0
    assert comparison["graph_resource_cost"] > 0.0
