"""Bounded reward-guided state graph runtime for CoRD."""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import torch


class CordOperator(str, Enum):
    CONTINUE = "continue"
    ROLLBACK = "rollback"
    BRANCH = "branch"
    MERGE = "merge"
    HALT = "halt"


@dataclass(frozen=True)
class CordSearchConfig:
    max_expansions: int = 8
    beam_size: int = 4
    exploration_weight: float = 0.2
    compute_cost_weight: float = 0.01
    memory_cost_weight: float = 0.01
    novelty_weight: float = 0.1
    deterministic: bool = True
    seed: int = 0
    max_verified_leaves: int = 4

    def __post_init__(self) -> None:
        if self.max_expansions < 0:
            raise ValueError("max_expansions must be non-negative")
        if self.beam_size < 1:
            raise ValueError("beam_size must be positive")
        if self.max_verified_leaves < 1:
            raise ValueError("max_verified_leaves must be positive")
        if self.exploration_weight < 0.0 or self.compute_cost_weight < 0.0:
            raise ValueError("exploration and compute-cost weights must be non-negative")
        if self.memory_cost_weight < 0.0 or self.novelty_weight < 0.0:
            raise ValueError("memory and novelty weights must be non-negative")


@dataclass(frozen=True)
class CordDecodeContext:
    """Prompt and decode budget supplied to a detached leaf decoder."""

    input_ids: torch.LongTensor
    prefix_lengths: torch.LongTensor
    max_new_tokens: int
    task_id: Optional[str] = None


@dataclass(frozen=True)
class CordVerifierResult:
    """Typed terminal evaluation; exact reward is kept separate from shaping."""

    valid: bool
    exact_reward: float
    shaping_reward: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reward(self) -> float:
        return self.exact_reward + self.shaping_reward


@dataclass
class CordStateRecord:
    state_id: str
    parent_ids: tuple[str, ...]
    operator: CordOperator
    depth: int
    value: float
    reward: float
    uncertainty: float
    novelty: float
    fingerprint: str
    rng_seed: int
    verified: bool = False
    provenance_ids: tuple[int, ...] = ()
    decoded_tokens: tuple[int, ...] = ()
    verifier_valid: Optional[bool] = None
    verifier_reason: str = ""
    resource_cost: float = 0.0
    state: Optional[torch.Tensor] = field(default=None, repr=False)
    checkpoint: Optional[torch.Tensor] = field(default=None, repr=False)


def state_fingerprint(state: torch.Tensor) -> str:
    """Create a stable, detached fingerprint for cycle detection."""

    normalized = torch.nn.functional.normalize(state.detach().float().reshape(-1), dim=0)
    quantized = (normalized.clamp(-4, 4) * 127).round().to(torch.int8).cpu().contiguous()
    return hashlib.sha1(quantized.numpy().tobytes()).hexdigest()


class CordStateArchive:
    """Archive with bounded GPU residency and CPU checkpoint spillover."""

    def __init__(self, hot_capacity: int, gpu_checkpoint_slots: int, checkpoint_interval: int):
        if hot_capacity < 1 or gpu_checkpoint_slots < 1 or checkpoint_interval < 1:
            raise ValueError("archive capacities and checkpoint_interval must be positive")
        self.hot_capacity = hot_capacity
        self.gpu_checkpoint_slots = gpu_checkpoint_slots
        self.checkpoint_interval = checkpoint_interval
        self.records: dict[str, CordStateRecord] = {}
        self.hot_state_ids: deque[str] = deque()
        self.gpu_checkpoints: deque[str] = deque(maxlen=gpu_checkpoint_slots)

    @property
    def hot_residency(self) -> int:
        return sum(self.records[state_id].state is not None for state_id in self.hot_state_ids)

    @property
    def checkpoint_residency(self) -> int:
        return sum(self.records[state_id].checkpoint is not None for state_id in self.gpu_checkpoints)

    @property
    def gpu_residency(self) -> int:
        resident_ids = set(self.hot_state_ids).union(self.gpu_checkpoints)
        return sum(self.records[state_id].state is not None for state_id in resident_ids)

    def add(self, record: CordStateRecord) -> None:
        self.records[record.state_id] = record
        self.hot_state_ids.append(record.state_id)
        if record.depth % self.checkpoint_interval == 0 and record.state is not None:
            record.checkpoint = (
                record.state.detach().to("cpu").pin_memory()
                if torch.cuda.is_available()
                else record.state.detach().cpu()
            )
            if len(self.gpu_checkpoints) == self.gpu_checkpoint_slots:
                evicted_checkpoint_id = self.gpu_checkpoints.popleft()
                if evicted_checkpoint_id not in self.hot_state_ids:
                    self.records[evicted_checkpoint_id].state = None
            self.gpu_checkpoints.append(record.state_id)
        self._evict_hot_states()

    def _evict_hot_states(self) -> None:
        while len(self.hot_state_ids) > self.hot_capacity:
            state_id = self.hot_state_ids.popleft()
            record = self.records[state_id]
            if state_id not in self.gpu_checkpoints:
                record.state = None

    def get(self, state_id: str) -> CordStateRecord:
        try:
            return self.records[state_id]
        except KeyError as error:
            raise KeyError(f"unknown CoRD state id: {state_id}") from error

    def materialize(
        self,
        state_id: str,
        replay: Callable[[torch.Tensor, CordStateRecord], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        record = self.get(state_id)
        if record.state is not None:
            return record.state.to(device)
        checkpoint_id = state_id
        while checkpoint_id not in self.gpu_checkpoints and self.records[checkpoint_id].checkpoint is None:
            parents = self.records[checkpoint_id].parent_ids
            if not parents:
                raise RuntimeError(f"state {state_id} has no replay checkpoint")
            checkpoint_id = parents[0]
        checkpoint_record = self.records[checkpoint_id]
        if checkpoint_record.state is not None:
            current = checkpoint_record.state.to(device)
        elif checkpoint_record.checkpoint is not None:
            current = checkpoint_record.checkpoint.to(device, non_blocking=True)
        else:
            raise RuntimeError(f"state {state_id} has no materializable checkpoint")
        target_id = state_id
        path: list[CordStateRecord] = []
        current_id = target_id
        while current_id != checkpoint_id:
            child = self.records[current_id]
            path.append(child)
            if not child.parent_ids:
                raise RuntimeError(f"state {target_id} has no path to its checkpoint")
            current_id = child.parent_ids[0]
        for child in reversed(path):
            current = replay(current, child)
        return current


__all__ = [
    "CordOperator",
    "CordSearchConfig",
    "CordStateRecord",
    "CordStateArchive",
    "state_fingerprint",
]
