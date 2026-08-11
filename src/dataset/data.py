"""ARC-AGI task discovery, prefix-causal samples, and batching helpers."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from .augment import apply_augmentation

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
GRID_OFFSET = 3
ROW_SEP = 13
IO_SEP = 14
PAIR_SEP = 15
ARC_VOCAB_SIZE = PAIR_SEP + 1
IGNORE_INDEX = -100

Grid = list[list[int]]
Pair = dict[str, Grid]


@dataclass(frozen=True)
class ARCSplit:
    """Auditable, task-level ARC training and validation partition."""

    train_files: tuple[Path, ...]
    validation_files: tuple[Path, ...]
    manifest_digest: str


def task_id_from_path(path: str | Path) -> str:
    """Return a canonical ARC task ID and reject ambiguous file names."""
    task_id = Path(path).stem
    if len(task_id) != 8 or any(character not in "0123456789abcdef" for character in task_id):
        raise ValueError(f"invalid ARC task filename: {path}")
    return task_id


def discover_arc_tasks(directory: str | Path) -> list[Path]:
    """Discover ARC JSON task files and reject duplicate IDs."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"ARC task directory does not exist: {directory}")
    files = sorted(directory.glob("*.json"), key=lambda path: task_id_from_path(path))
    if not files:
        raise ValueError(f"no ARC JSON task files found in {directory}")
    task_ids = [task_id_from_path(path) for path in files]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate ARC task IDs in {directory}")
    return files


def split_arc_training_files(
    file_paths: Sequence[str | Path], *, seed: int = 42, train_fraction: float = 0.8
) -> ARCSplit:
    """Create a deterministic task-level train/validation split independent of path order."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between zero and one")
    paths_by_id = {task_id_from_path(path): Path(path) for path in file_paths}
    if len(paths_by_id) != len(file_paths):
        raise ValueError("ARC training files must have unique task IDs")
    ranked_ids = sorted(
        paths_by_id,
        key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest(),
    )
    split_index = int(len(ranked_ids) * train_fraction)
    if split_index == 0 or split_index == len(ranked_ids):
        raise ValueError("split leaves an empty train or validation partition")
    train_ids = ranked_ids[:split_index]
    validation_ids = ranked_ids[split_index:]
    manifest = "\n".join(
        [f"seed={seed}", f"train_fraction={train_fraction:.12g}", *train_ids, "--validation--", *validation_ids]
    )
    return ARCSplit(
        train_files=tuple(paths_by_id[task_id] for task_id in train_ids),
        validation_files=tuple(paths_by_id[task_id] for task_id in validation_ids),
        manifest_digest=hashlib.sha256(manifest.encode("utf-8")).hexdigest(),
    )


def _validate_grid(grid: Any, *, task_id: str, pair_kind: str, pair_index: int) -> Grid:
    context = f"task {task_id} {pair_kind}[{pair_index}]"
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) and row for row in grid):
        raise ValueError(f"{context} must be a nonempty two-dimensional grid")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError(f"{context} grid must be rectangular")
    if any(not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 9 for row in grid for value in row):
        raise ValueError(f"{context} colors must be integers in [0, 9]")
    return [row[:] for row in grid]


def load_arc_task(path: str | Path) -> dict[str, list[Pair]]:
    """Load and strictly validate a public ARC JSON task."""
    path = Path(path)
    task_id = task_id_from_path(path)
    with path.open("r", encoding="utf-8") as file:
        task = json.load(file)
    if not isinstance(task, dict) or not isinstance(task.get("train"), list) or not isinstance(task.get("test"), list):
        raise ValueError(f"task {task_id} must contain list-valued train and test fields")
    if not task["train"] or not task["test"]:
        raise ValueError(f"task {task_id} must contain at least one demonstration and query")
    validated: dict[str, list[Pair]] = {"train": [], "test": []}
    for pair_kind in ("train", "test"):
        for pair_index, pair in enumerate(task[pair_kind]):
            if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
                raise ValueError(f"task {task_id} {pair_kind}[{pair_index}] must contain input and output")
            validated[pair_kind].append(
                {
                    "input": _validate_grid(pair["input"], task_id=task_id, pair_kind=f"{pair_kind}.input", pair_index=pair_index),
                    "output": _validate_grid(pair["output"], task_id=task_id, pair_kind=f"{pair_kind}.output", pair_index=pair_index),
                }
            )
    return validated


def _grid_tokens(grid: Grid) -> list[int]:
    tokens: list[int] = []
    for row in grid:
        tokens.extend(value + GRID_OFFSET for value in row)
        tokens.append(ROW_SEP)
    return tokens


def _pair_prefix_tokens(pair: Pair, *, add_separator: bool) -> list[int]:
    tokens = _grid_tokens(pair["input"])
    tokens.append(IO_SEP)
    tokens.extend(_grid_tokens(pair["output"]))
    if add_separator:
        tokens.append(PAIR_SEP)
    return tokens


class ARCDataset(Dataset):
    """Prefix-causal ARC queries with optional task-consistent train variants.

    ``num_aug`` is the number of additional augmented copies per query in each
    epoch.  Copy zero is always canonical, which keeps a non-augmented anchor
    in the train distribution and makes validation/test protocol-independent.
    """

    def __init__(
        self,
        file_paths: Sequence[str | Path],
        *,
        max_length: int | None = None,
        augment: bool = False,
        num_aug: int = 0,
        augmentation_seed: int = 42,
        split_name: str = "train",
    ):
        if split_name not in {"train", "validation", "test"}:
            raise ValueError("split_name must be train, validation, or test")
        if num_aug < 0:
            raise ValueError("num_aug must be non-negative")
        if num_aug and split_name != "train":
            raise ValueError("num_aug is supported only for the train split")
        self.file_paths = tuple(Path(path) for path in file_paths)
        self.max_length = max_length
        self.augment = augment
        self.num_aug = num_aug
        self.augmentation_seed = augmentation_seed
        self.split_name = split_name
        self.epoch = 0
        self.tasks = [(task_id_from_path(path), load_arc_task(path)) for path in self.file_paths]
        self.samples = [
            (task_id, task, query_index, augmentation_index)
            for task_id, task in self.tasks
            for query_index in range(len(task["test"]))
            for augmentation_index in range(1 + (num_aug if augment and split_name == "train" else 0))
        ]
        self._validate_lengths()

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic epoch component used for train augmentation."""
        self.epoch = epoch

    def _task_pairs(
        self,
        task_id: str,
        task: dict[str, list[Pair]],
        augmentation_index: int = 0,
    ) -> tuple[list[Pair], list[Pair]]:
        pairs = [*task["train"], *task["test"]]
        if not self.augment or augmentation_index == 0:
            return task["train"], task["test"]
        if self.split_name != "train" or augmentation_index > self.num_aug:
            raise ValueError("invalid ARC augmentation index")
        epoch = self.epoch if self.split_name == "train" else 0
        digest = hashlib.sha256(
            f"{self.augmentation_seed}:{self.split_name}:{epoch}:{task_id}:{augmentation_index}".encode("utf-8")
        ).digest()
        augmented = apply_augmentation(pairs, rng=random.Random(int.from_bytes(digest[:8], "big")))
        train_count = len(task["train"])
        return augmented[:train_count], augmented[train_count:]

    @staticmethod
    def _make_sample(
        task_id: str,
        demonstrations: Sequence[Pair],
        query: Pair,
        query_index: int,
        augmentation_index: int = 0,
    ) -> dict[str, Any]:
        prefix = [BOS_TOKEN]
        for demo_index, demonstration in enumerate(demonstrations):
            prefix.extend(_pair_prefix_tokens(demonstration, add_separator=True))
        prefix.extend(_grid_tokens(query["input"]))
        prefix.append(IO_SEP)
        target = [*_grid_tokens(query["output"]), EOS_TOKEN]
        input_ids = [*prefix, *target]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor([*[IGNORE_INDEX] * len(prefix), *target], dtype=torch.long),
            "prefix_length": len(prefix),
            "target_ids": torch.tensor(target, dtype=torch.long),
            "target_grid": [row[:] for row in query["output"]],
            "task_id": task_id,
            "query_index": query_index,
            "augmentation_index": augmentation_index,
        }

    def _validate_lengths(self) -> None:
        if self.max_length is None:
            return
        overlong = []
        for task_id, task, query_index, augmentation_index in self.samples:
            demonstrations, queries = self._task_pairs(task_id, task, augmentation_index)
            sample = self._make_sample(
                task_id, demonstrations, queries[query_index], query_index, augmentation_index
            )
            if sample["input_ids"].numel() > self.max_length:
                overlong.append(f"{task_id}[{query_index}]={sample['input_ids'].numel()}")
        if overlong:
            raise ValueError(
                f"ARC samples exceed configured max_length={self.max_length}; no tokens were truncated: "
                + ", ".join(overlong[:10])
            )

    def lengths(self) -> dict[str, int]:
        values = []
        for index in range(len(self)):
            sample = self[index]
            values.append((sample["prefix_length"], sample["target_ids"].numel(), sample["input_ids"].numel()))
        if not values:
            return {"samples": 0, "max_prefix": 0, "max_target": 0, "max_total": 0}
        return {
            "samples": len(values),
            "max_prefix": max(value[0] for value in values),
            "max_target": max(value[1] for value in values),
            "max_total": max(value[2] for value in values),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task_id, task, query_index, augmentation_index = self.samples[index]
        demonstrations, queries = self._task_pairs(task_id, task, augmentation_index)
        return self._make_sample(task_id, demonstrations, queries[query_index], query_index, augmentation_index)


def collate_fn(batch: Sequence[dict[str, Any]], max_length: int | None = None) -> dict[str, Any]:
    """Dynamically pad prefix-causal ARC samples without truncating targets."""
    if not batch:
        raise ValueError("cannot collate an empty batch")
    batch_max_length = max(item["input_ids"].numel() for item in batch)
    if max_length is not None and batch_max_length > max_length:
        raise ValueError(f"batch length {batch_max_length} exceeds configured max_length={max_length}; refusing truncation")
    input_ids, attention_masks, labels, target_ids, prefix_lengths = [], [], [], [], []
    max_target_length = max(item["target_ids"].numel() for item in batch)
    for item in batch:
        sequence_length = item["input_ids"].numel()
        padding = batch_max_length - sequence_length
        input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, padding), value=PAD_TOKEN))
        attention_masks.append(torch.nn.functional.pad(item["attention_mask"], (0, padding), value=0))
        labels.append(torch.nn.functional.pad(item["labels"], (0, padding), value=IGNORE_INDEX))
        target_ids.append(torch.nn.functional.pad(item["target_ids"], (0, max_target_length - item["target_ids"].numel()), value=IGNORE_INDEX))
        prefix_lengths.append(item["prefix_length"])
    result = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "labels": torch.stack(labels),
        "target_ids": torch.stack(target_ids),
        "prefix_lengths": torch.tensor(prefix_lengths, dtype=torch.long),
        "target_grids": [item["target_grid"] for item in batch],
        "task_ids": [item["task_id"] for item in batch],
        "query_indices": [item["query_index"] for item in batch],
    }
    for row, prefix_length in enumerate(prefix_lengths):
        if result["labels"][row, :prefix_length].ne(IGNORE_INDEX).any():
            raise ValueError("prefix tokens must not be supervised")
        valid_labels = result["labels"][row].ne(IGNORE_INDEX).nonzero(as_tuple=False).flatten()
        if not len(valid_labels) or valid_labels[0].item() != prefix_length:
            raise ValueError("first supervised token must begin at prefix_length")
        if result["labels"][row, result["attention_mask"][row].eq(0)].ne(IGNORE_INDEX).any():
            raise ValueError("padding positions must not be supervised")
    return result


__all__ = [
    "ARC_VOCAB_SIZE", "ARCDataset", "ARCSplit", "BOS_TOKEN", "EOS_TOKEN", "GRID_OFFSET", "IGNORE_INDEX",
    "IO_SEP", "PAD_TOKEN", "PAIR_SEP", "ROW_SEP", "collate_fn", "discover_arc_tasks", "load_arc_task",
    "split_arc_training_files", "task_id_from_path",
]
