import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from dataset.data import (
    EOS_TOKEN,
    IGNORE_INDEX,
    IO_SEP,
    ARCDataset,
    collate_fn,
    discover_arc_tasks,
    split_arc_training_files,
)


def write_task(path: Path, *, queries=1):
    task = {
        "train": [{"input": [[1, 0]], "output": [[0, 1]]}],
        "test": [{"input": [[2]], "output": [[3, 3]]} for _ in range(queries)],
    }
    path.write_text(json.dumps(task), encoding="utf-8")


def test_deterministic_task_split_is_order_independent(tmp_path):
    paths = []
    for value in range(10):
        path = tmp_path / f"{value:08x}.json"
        write_task(path)
        paths.append(path)
    first = split_arc_training_files(paths, seed=7)
    second = split_arc_training_files(list(reversed(paths)), seed=7)
    assert first == second
    assert len(first.train_files) == 8
    assert len(first.validation_files) == 2
    assert not set(first.train_files) & set(first.validation_files)


def test_prefix_causal_queries_and_collation(tmp_path):
    path = tmp_path / "deadbeef.json"
    write_task(path, queries=2)
    dataset = ARCDataset([path], split_name="train")
    assert len(dataset) == 2
    sample = dataset[0]
    prefix = sample["prefix_length"]
    assert sample["labels"][:prefix].eq(IGNORE_INDEX).all()
    assert sample["labels"][prefix:].equal(sample["target_ids"])
    assert sample["input_ids"][prefix - 1].item() == IO_SEP
    assert sample["target_ids"][-1].item() == EOS_TOKEN
    batch = collate_fn([sample, dataset[1]])
    assert batch["prefix_lengths"].tolist() == [prefix, prefix]
    assert batch["labels"][:, :prefix].eq(IGNORE_INDEX).all()


def test_train_num_aug_keeps_canonical_and_adds_deterministic_task_variants(tmp_path):
    path = tmp_path / "deadbeef.json"
    write_task(path)
    train = ARCDataset([path], augment=True, num_aug=2, augmentation_seed=3, split_name="train")
    assert len(train) == 3
    canonical, first_variant, second_variant = (train[index] for index in range(3))
    assert [sample["augmentation_index"] for sample in (canonical, first_variant, second_variant)] == [0, 1, 2]
    assert canonical["task_id"] == first_variant["task_id"] == second_variant["task_id"]
    assert canonical["query_index"] == first_variant["query_index"] == second_variant["query_index"]
    assert not torch.equal(canonical["input_ids"], first_variant["input_ids"])
    first = first_variant["input_ids"]
    train.set_epoch(1)
    second = train[1]["input_ids"]
    assert not torch.equal(first, second)


def test_num_aug_is_train_only_and_must_be_nonnegative(tmp_path):
    path = tmp_path / "deadbeef.json"
    write_task(path)
    with pytest.raises(ValueError, match="non-negative"):
        ARCDataset([path], num_aug=-1)
    with pytest.raises(ValueError, match="train split"):
        ARCDataset([path], num_aug=1, split_name="validation")


def test_invalid_grid_and_length_are_rejected(tmp_path):
    path = tmp_path / "deadbeef.json"
    path.write_text(json.dumps({"train": [{"input": [[10]], "output": [[0]]}], "test": [{"input": [[0]], "output": [[0]]}]}))
    with pytest.raises(ValueError, match="colors"):
        ARCDataset([path])
    write_task(path)
    with pytest.raises(ValueError, match="no tokens were truncated"):
        ARCDataset([path], max_length=1)


def test_discovery_rejects_non_arc_filenames(tmp_path):
    write_task(tmp_path / "deadbeef.json")
    (tmp_path / "not-a-task.json").write_text("{}")
    with pytest.raises(ValueError, match="invalid ARC task filename"):
        discover_arc_tasks(tmp_path)
