"""Offline ARC terminal verifier used by detached CoRD graph search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cord.state_graph import CordVerifierResult

from .trainer import decode_arc_grid


@dataclass(frozen=True)
class ARCDecodedVerifier:
    """Evaluate one decoded completion against its known offline ARC target."""

    target_grid: list[list[int]]
    shaping: bool = False

    def __call__(self, tokens: Iterable[int], _context: object = None) -> CordVerifierResult:
        decoded = decode_arc_grid(tokens)
        if decoded is None:
            return CordVerifierResult(False, 0.0, reason="malformed_arc_completion")
        exact = decoded == self.target_grid
        shaping_reward = 0.0
        if self.shaping and not exact and len(decoded) == len(self.target_grid) and all(
            len(row) == len(target_row) for row, target_row in zip(decoded, self.target_grid)
        ):
            total = sum(len(row) for row in self.target_grid)
            shaping_reward = sum(
                value == target_value
                for row, target_row in zip(decoded, self.target_grid)
                for value, target_value in zip(row, target_row)
            ) / total
        return CordVerifierResult(
            valid=True,
            exact_reward=float(exact),
            shaping_reward=shaping_reward,
            reason="exact" if exact else "wrong_grid",
            metadata={"decoded_grid": decoded},
        )


__all__ = ["ARCDecodedVerifier"]
