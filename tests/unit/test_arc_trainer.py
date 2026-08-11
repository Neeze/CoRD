import pytest

torch = pytest.importorskip("torch")

from dataset.data import EOS_TOKEN, GRID_OFFSET, ROW_SEP
from trainer.trainer import ARCMetrics, decode_arc_grid


def test_decode_arc_grid_and_metrics():
    tokens = [GRID_OFFSET + 2, GRID_OFFSET + 3, ROW_SEP, EOS_TOKEN]
    assert decode_arc_grid(tokens) == [[2, 3]]
    assert decode_arc_grid([GRID_OFFSET + 1, EOS_TOKEN]) is None
    metrics = ARCMetrics()
    logits = torch.full((1, len(tokens), 32), -100.0)
    target = torch.tensor([tokens])
    logits[0, torch.arange(len(tokens)), target[0]] = 100.0
    metrics.update_teacher_forced(logits, target)
    metrics.update_generated("task-one", tokens, [[2, 3]])
    values = metrics.as_dict()
    assert values["token_accuracy"] == 1.0
    assert values["completion_exact"] == 1.0
    assert values["grid_exact"] == 1.0
    assert values["task_exact"] == 1.0


def test_malformed_generation_is_counted():
    metrics = ARCMetrics()
    metrics.examples = 1
    metrics.update_generated("task", [99, EOS_TOKEN], [[0]])
    values = metrics.as_dict()
    assert values["malformed_outputs"] == 1.0
    assert values["valid_grid_rate"] == 0.0
