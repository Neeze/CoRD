"""Deterministic, task-consistent ARC augmentations."""

from __future__ import annotations

import random
from collections.abc import Sequence


def dihedral_transform(grid: Sequence[Sequence[int]], trans_id: int) -> list[list[int]]:
    """Apply one of the eight dihedral transformations to a two-dimensional grid."""
    if not grid:
        return []
    if trans_id == 0:
        return [list(row) for row in grid]
    if trans_id == 1:
        return [list(row) for row in zip(*grid[::-1])]
    if trans_id == 2:
        return [list(row)[::-1] for row in grid[::-1]]
    if trans_id == 3:
        return [list(row) for row in zip(*grid)][::-1]
    if trans_id == 4:
        return [list(row) for row in grid[::-1]]
    if trans_id == 5:
        return [list(row)[::-1] for row in grid]
    if trans_id == 6:
        return [list(row) for row in zip(*grid)]
    if trans_id == 7:
        return [list(row)[::-1] for row in zip(*grid[::-1])]
    raise ValueError(f"unknown dihedral transform: {trans_id}")


def augmentation_descriptor(rng: random.Random) -> tuple[list[int], int]:
    """Sample a color permutation and spatial transform from an explicit RNG."""
    return [0, *rng.sample(range(1, 10), 9)], rng.randrange(8)


def apply_augmentation(
    task_pairs: Sequence[dict[str, list[list[int]]]],
    *,
    rng: random.Random | None = None,
    descriptor: tuple[Sequence[int], int] | None = None,
) -> list[dict[str, list[list[int]]]]:
    """Augment every input/output pair with one shared task-level transform.

    Color zero is preserved; all nonzero colors are permuted. Supplying an RNG or
    a descriptor makes augmentation reproducible and avoids global random state.
    """
    if descriptor is None:
        descriptor = augmentation_descriptor(rng if rng is not None else random.Random())
    color_map, trans_id = descriptor
    if len(color_map) != 10 or color_map[0] != 0:
        raise ValueError("augmentation color map must preserve color zero")

    augmented = []
    for pair in task_pairs:
        augmented.append(
            {
                "input": dihedral_transform(
                    [[color_map[value] for value in row] for row in pair["input"]], trans_id
                ),
                "output": dihedral_transform(
                    [[color_map[value] for value in row] for row in pair["output"]], trans_id
                ),
            }
        )
    return augmented
