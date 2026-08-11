import pytest

torch = pytest.importorskip("torch")

from cord.state_graph import (
    CordDecodeContext,
    CordOperator,
    CordSearchConfig,
    CordStateArchive,
    CordStateRecord,
    CordVerifierResult,
    state_fingerprint,
)
from trainer.arc_verifier import ARCDecodedVerifier
from dataset.data import EOS_TOKEN, GRID_OFFSET, ROW_SEP


def test_archive_bounds_hot_residency():
    archive = CordStateArchive(hot_capacity=4, gpu_checkpoint_slots=2, checkpoint_interval=4)
    parent_id = "root"
    root = torch.zeros(2, 4)
    archive.add(
        CordStateRecord(
            "root", (), CordOperator.CONTINUE, 0, 0.0, 0.0, 0.0, 0.0,
            state_fingerprint(root), 0, state=root,
        )
    )
    for depth in range(1, 129):
        state = torch.full((2, 4), float(depth))
        state_id = f"state-{depth}"
        archive.add(
            CordStateRecord(
                state_id, (parent_id,), CordOperator.CONTINUE, depth, 0.0, 0.0,
                0.0, 0.0, state_fingerprint(state), depth, state=state,
            )
        )
        parent_id = state_id
    assert archive.hot_residency <= 4
    assert archive.checkpoint_residency <= 2
    assert archive.gpu_residency <= 6


def test_archive_replay_from_cpu_checkpoint():
    archive = CordStateArchive(hot_capacity=1, gpu_checkpoint_slots=1, checkpoint_interval=2)
    root = torch.zeros(1, 2)
    archive.add(
        CordStateRecord(
            "root", (), CordOperator.CONTINUE, 0, 0.0, 0.0, 0.0, 0.0,
            state_fingerprint(root), 0, state=root,
        )
    )
    child = torch.ones(1, 2)
    archive.add(
        CordStateRecord(
            "child", ("root",), CordOperator.CONTINUE, 1, 0.0, 0.0, 0.0,
            0.0, state_fingerprint(child), 1, state=child,
        )
    )
    grandchild = torch.full((1, 2), 2.0)
    archive.add(
        CordStateRecord(
            "grandchild", ("child",), CordOperator.CONTINUE, 2, 0.0, 0.0,
            0.0, 0.0, state_fingerprint(grandchild), 2, state=grandchild,
        )
    )
    replayed = archive.materialize("child", lambda state, record: state + 1, torch.device("cpu"))
    assert torch.equal(replayed, child)


def test_archive_releases_evicted_gpu_checkpoint_state():
    archive = CordStateArchive(hot_capacity=1, gpu_checkpoint_slots=1, checkpoint_interval=1)
    root = torch.zeros(1, 2)
    child = torch.ones(1, 2)
    archive.add(
        CordStateRecord(
            "root", (), CordOperator.CONTINUE, 0, 0.0, 0.0, 0.0, 0.0,
            state_fingerprint(root), 0, state=root,
        )
    )
    archive.add(
        CordStateRecord(
            "child", ("root",), CordOperator.CONTINUE, 1, 0.0, 0.0, 0.0, 0.0,
            state_fingerprint(child), 1, state=child,
        )
    )
    assert archive.records["root"].state is None
    assert archive.records["root"].checkpoint is not None


def test_arc_decoded_verifier_rejects_malformed_and_separates_shaping():
    verifier = ARCDecodedVerifier([[2]], shaping=True)
    malformed = verifier([99, EOS_TOKEN])
    wrong = verifier([GRID_OFFSET + 1, ROW_SEP, EOS_TOKEN])
    exact = verifier([GRID_OFFSET + 2, ROW_SEP, EOS_TOKEN])
    assert not malformed.valid and malformed.exact_reward == 0.0
    assert wrong.valid and wrong.exact_reward == 0.0 and wrong.shaping_reward == 0.0
    assert exact.valid and exact.exact_reward == 1.0


def test_search_decodes_and_verifies_multiple_terminal_leaves():
    from cord import CordConfig, CordModel

    config = CordConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_decoder_layers=1, num_attention_heads=4, num_key_value_heads=2,
        concept_slots=2, concept_num_attention_heads=4, concept_latent_size=8,
        num_recurrent_loops=1, minimum_recurrent_loops=1, max_recurrent_loops=1,
        num_experts=1, num_experts_per_token=1, routed_latent_width=8,
        expert_intermediate_size=16, max_position_embeddings=16,
    )
    model = CordModel(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    context = CordDecodeContext(prompt, torch.tensor([4]), max_new_tokens=3, task_id="task")
    calls = 0

    def decode(_state, _context):
        nonlocal calls
        calls += 1
        token = GRID_OFFSET + 1 if calls == 1 else GRID_OFFSET + 2
        return [token, ROW_SEP, EOS_TOKEN]

    result = model.search(
        prompt,
        prefix_lengths=torch.tensor([4]),
        search_config=CordSearchConfig(max_expansions=1, beam_size=2, max_verified_leaves=2),
        decode_context=context,
        decode_state=decode,
        verifier=ARCDecodedVerifier([[2]]),
    )
    assert calls == 2
    assert result.verified is not None and any(result.verified)
    assert result.decoded_tokens is not None
    assert result.rewards is not None and result.rewards.max().item() == 1.0
    assert result.best_state_id == result.state_ids[1]
    assert result.trajectory is not None and all("resource_cost" in record for record in result.trajectory)
    assert all(record["task_id"] == "task" for record in result.trajectory)


def test_search_trajectory_records_merge_parents_and_replays_deterministically():
    from cord import CordConfig, CordModel

    torch.manual_seed(7)
    config = CordConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_decoder_layers=1, num_attention_heads=4, num_key_value_heads=2,
        concept_slots=2, concept_num_attention_heads=4, concept_latent_size=8,
        num_recurrent_loops=1, minimum_recurrent_loops=1, max_recurrent_loops=1,
        num_experts=1, num_experts_per_token=1, routed_latent_width=8,
        expert_intermediate_size=16, max_position_embeddings=16,
    )
    model = CordModel(config).eval()
    prompt = torch.randint(0, config.vocab_size, (1, 4))
    context = CordDecodeContext(prompt, torch.tensor([4]), max_new_tokens=3, task_id="replay")

    def decode(_state, _context):
        return [GRID_OFFSET + 1, ROW_SEP, EOS_TOKEN]

    settings = CordSearchConfig(max_expansions=2, beam_size=8, max_verified_leaves=2, seed=11)
    first = model.search(prompt, torch.tensor([4]), settings, context, decode, ARCDecodedVerifier([[1]]))
    second = model.search(prompt, torch.tensor([4]), settings, context, decode, ARCDecodedVerifier([[1]]))
    assert first.trajectory == second.trajectory
    assert all("parent_ids" in record and "prompt_identity" in record for record in first.trajectory)


def test_merge_operator_uses_other_concept_packet():
    from cord import CordConfig, CordModel

    config = CordConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_decoder_layers=1, num_attention_heads=4, num_key_value_heads=2,
        concept_slots=2, concept_num_attention_heads=4, concept_latent_size=8,
        num_recurrent_loops=1, minimum_recurrent_loops=1, max_recurrent_loops=1,
        num_experts=1, num_experts_per_token=1, routed_latent_width=8,
        expert_intermediate_size=16, max_position_embeddings=16,
    )
    model = CordModel(config).eval()
    state = torch.randn(2, config.hidden_size)
    left_parent = torch.zeros_like(state)
    right_parent = torch.ones_like(state)
    left = model._operator_state(state, CordOperator.MERGE, left_parent)
    right = model._operator_state(state, CordOperator.MERGE, right_parent)
    assert not torch.allclose(left, right)
