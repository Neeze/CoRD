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
    discount: float = 0.99
    cycle_penalty: float = 0.1
    decode_cost_weight: float = 0.001
    controller_temperature: float = 1.0
    shaping_reward_clip: float = 0.2
    valid_reward: float = 0.01

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
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between zero and one")
        if self.cycle_penalty < 0.0 or self.decode_cost_weight < 0.0:
            raise ValueError("cycle and decode-cost weights must be non-negative")
        if self.controller_temperature <= 0.0:
            raise ValueError("controller_temperature must be positive")
        if self.shaping_reward_clip < 0.0:
            raise ValueError("shaping_reward_clip must be non-negative")
        if self.valid_reward < 0.0:
            raise ValueError("valid_reward must be non-negative")


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
    action_parent_id: Optional[str] = None
    second_parent_id: Optional[str] = None
    policy_log_prob: float = 0.0
    selection_score: float = 0.0
    return_value: float = 0.0
    advantage: float = 0.0
    exact_reward: float = 0.0
    shaping_reward: float = 0.0
    terminal: bool = False
    cycle_detected: bool = False


def state_fingerprint(state: torch.Tensor) -> str:
    """Create a stable, detached fingerprint for cycle detection."""

    normalized = torch.nn.functional.normalize(state.detach().float().reshape(-1), dim=0)
    quantized = (normalized.clamp(-4, 4) * 127).round().to(torch.int8).cpu().contiguous()
    return hashlib.sha1(quantized.numpy().tobytes()).hexdigest()


def state_novelty(state: torch.Tensor, archive_states: list[torch.Tensor]) -> float:
    """Cosine-distance novelty against materialized archived concept packets."""

    if not archive_states:
        return 1.0
    candidate = torch.nn.functional.normalize(state.detach().float().reshape(-1), dim=0)
    similarities = []
    for archived in archive_states:
        reference = torch.nn.functional.normalize(archived.detach().float().reshape(-1), dim=0)
        similarities.append(torch.dot(candidate, reference).clamp(-1.0, 1.0))
    maximum_similarity = torch.stack(similarities).max()
    return float(((1.0 - maximum_similarity) * 0.5).clamp(0.0, 1.0).item())


def backup_graph_returns(
    records: dict[str, CordStateRecord],
    *,
    discount: float = 0.99,
    compute_cost_weight: float = 0.0,
    decode_cost_weight: float = 0.0,
) -> None:
    """Monte-Carlo/max backup of verified terminal returns through a latent DAG.

    The update is deliberately offline: terminal verification is completed
    before any result is propagated, so the verifier never becomes an action
    selector during the episode.
    """

    children: dict[str, list[CordStateRecord]] = {state_id: [] for state_id in records}
    for record in records.values():
        for parent_id in record.parent_ids:
            if parent_id in children:
                children[parent_id].append(record)
    ordered = sorted(records.values(), key=lambda item: item.depth, reverse=True)
    for record in ordered:
        if record.terminal:
            terminal_reward = record.exact_reward + record.shaping_reward
            terminal_reward -= decode_cost_weight * len(record.decoded_tokens)
            record.return_value = terminal_reward
        elif children[record.state_id]:
            record.return_value = max(
                discount * child.return_value - compute_cost_weight for child in children[record.state_id]
            )
        else:
            record.return_value = -compute_cost_weight
        record.advantage = record.return_value - record.value


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
    "CordDecodeContext",
    "CordVerifierResult",
    "CordStateRecord",
    "CordStateArchive",
    "state_fingerprint",
    "state_novelty",
    "backup_graph_returns",
]
