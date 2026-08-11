import pytest

torch = pytest.importorskip("torch")

from cord.state_graph import CordOperator, CordStateArchive, CordStateRecord, state_fingerprint


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
