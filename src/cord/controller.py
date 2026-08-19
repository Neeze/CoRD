"""Learned hierarchical controller for recurrent concept-state graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .state_graph import CordOperator


@dataclass
class CordControllerOutput:
    """Dense scores for the factorized graph action distribution.

    The action distribution is
    ``p(parent) p(operator | parent) p(parent2 | parent, operator=merge)``.
    Values and uncertainties are state estimates for every archived parent.
    """

    parent_logits: torch.Tensor
    operator_logits: torch.Tensor
    second_parent_logits: torch.Tensor
    values: torch.Tensor
    uncertainties: torch.Tensor


class CordStateController(nn.Module):
    """Select an archived state, structural operator, and optional merge peer."""

    def __init__(
        self,
        hidden_size: int,
        num_operators: int,
        dropout: float = 0.0,
        controller_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        if num_operators != len(CordOperator):
            raise ValueError("controller operator count must match CordOperator")
        self.hidden_size = hidden_size
        self.num_operators = num_operators
        controller_hidden_size = controller_hidden_size or min(hidden_size, max(16, hidden_size // 8))
        self.controller_hidden_size = controller_hidden_size
        self.state_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.parent_selector = nn.Linear(hidden_size, controller_hidden_size, bias=False)
        self.parent_query = nn.Linear(hidden_size, controller_hidden_size, bias=False)
        self.operator_policy = nn.Sequential(
            nn.Linear(hidden_size * 2, controller_hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(controller_hidden_size, num_operators),
        )
        self.second_parent_selector = nn.Linear(hidden_size, controller_hidden_size, bias=False)
        self.second_parent_query = nn.Linear(hidden_size * 2, controller_hidden_size, bias=False)
        self.budget_projection = nn.Linear(1, hidden_size, bias=False)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size * 2, controller_hidden_size), nn.SiLU(), nn.Linear(controller_hidden_size, 1)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size * 2, controller_hidden_size), nn.SiLU(), nn.Linear(controller_hidden_size, 1)
        )

    @staticmethod
    def _pool(states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 2:
            return states.mean(dim=0, keepdim=True)
        if states.ndim == 3:
            return states.mean(dim=1)
        if states.ndim == 4:
            return states.mean(dim=2)
        raise ValueError("concept states must have shape [slots, hidden], [states, slots, hidden], or batched")

    def forward(
        self,
        current_state: torch.Tensor,
        archive_states: torch.Tensor,
        goal_state: torch.Tensor,
        budget_remaining: torch.Tensor | float,
        *,
        archive_mask: Optional[torch.Tensor] = None,
    ) -> CordControllerOutput:
        """Score a possibly padded archive.

        Unbatched inputs are promoted to a batch of one.  Returned tensors keep
        the batch dimension, which makes training on padded replay batches and
        inference on a single graph use the exact same policy.
        """

        if archive_states.ndim == 3:
            archive_states = archive_states.unsqueeze(0)
        if current_state.ndim == 2:
            current_state = current_state.unsqueeze(0)
        if goal_state.ndim == 2:
            goal_state = goal_state.unsqueeze(0)
        if archive_states.ndim != 4 or current_state.ndim != 3 or goal_state.ndim != 3:
            raise ValueError("invalid controller state rank")
        batch_size, archive_size, _, hidden_size = archive_states.shape
        if hidden_size != self.hidden_size:
            raise ValueError("controller state hidden size does not match its configuration")
        if current_state.shape[0] != batch_size or goal_state.shape[0] != batch_size:
            raise ValueError("controller inputs must share a batch dimension")

        archive = self.state_norm(self._pool(archive_states))
        current = self.context_norm(self._pool(current_state))
        goal = self.context_norm(self._pool(goal_state))
        if not torch.is_tensor(budget_remaining):
            budget_remaining = archive.new_full((batch_size,), float(budget_remaining))
        budget = budget_remaining.to(device=archive.device, dtype=archive.dtype).reshape(batch_size, 1)
        context = current + goal + self.budget_projection(budget)

        parent_keys = self.parent_selector(archive)
        parent_query = self.parent_query(context).unsqueeze(1)
        parent_logits = (parent_keys * parent_query).sum(dim=-1) / self.controller_hidden_size**0.5
        pair_context = torch.cat((archive, context.unsqueeze(1).expand(-1, archive_size, -1)), dim=-1)
        operator_logits = self.operator_policy(pair_context)

        second_keys = self.second_parent_selector(archive)
        second_queries = self.second_parent_query(pair_context)
        second_parent_logits = (
            torch.einsum("bih,bjh->bij", second_queries, second_keys) / self.controller_hidden_size**0.5
        )
        diagonal = torch.eye(archive_size, device=archive.device, dtype=torch.bool).unsqueeze(0)
        second_parent_logits = second_parent_logits.masked_fill(diagonal, torch.finfo(archive.dtype).min)

        values = self.value_head(pair_context).squeeze(-1)
        uncertainties = F.softplus(self.uncertainty_head(pair_context).squeeze(-1)) + 1e-6
        if archive_mask is not None:
            mask = archive_mask.to(device=archive.device, dtype=torch.bool)
            if mask.shape != (batch_size, archive_size):
                raise ValueError("archive_mask must have shape [batch, archive]")
            minimum = torch.finfo(archive.dtype).min
            parent_logits = parent_logits.masked_fill(~mask, minimum)
            operator_logits = operator_logits.masked_fill(~mask.unsqueeze(-1), minimum)
            second_parent_logits = second_parent_logits.masked_fill(~mask.unsqueeze(1), minimum)
        return CordControllerOutput(
            parent_logits=parent_logits,
            operator_logits=operator_logits,
            second_parent_logits=second_parent_logits,
            values=values,
            uncertainties=uncertainties,
        )

    @staticmethod
    def action_log_prob(
        output: CordControllerOutput,
        parent_index: torch.Tensor,
        operator_index: torch.Tensor,
        second_parent_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the hierarchical log probability of replayed actions."""

        batch = torch.arange(output.parent_logits.shape[0], device=output.parent_logits.device)
        parent_log_prob = F.log_softmax(output.parent_logits, dim=-1)[batch, parent_index]
        operator_log_prob = F.log_softmax(output.operator_logits[batch, parent_index], dim=-1)[
            batch, operator_index
        ]
        log_prob = parent_log_prob + operator_log_prob
        merge_index = list(CordOperator).index(CordOperator.MERGE)
        merge_mask = operator_index.eq(merge_index)
        if merge_mask.any():
            if second_parent_index is None:
                raise ValueError("merge actions require second_parent_index")
            second_logits = output.second_parent_logits[batch, parent_index]
            second_log_prob = F.log_softmax(second_logits, dim=-1)[batch, second_parent_index]
            log_prob = log_prob + torch.where(merge_mask, second_log_prob, torch.zeros_like(second_log_prob))
        return log_prob

    @staticmethod
    def entropy(output: CordControllerOutput) -> torch.Tensor:
        """Factor entropy used to prevent early controller collapse."""

        parent_probs = F.softmax(output.parent_logits, dim=-1)
        parent_entropy = -(parent_probs * F.log_softmax(output.parent_logits, dim=-1)).sum(dim=-1)
        operator_log_probs = F.log_softmax(output.operator_logits, dim=-1)
        operator_entropy = -(operator_log_probs.exp() * operator_log_probs).sum(dim=-1)
        return parent_entropy + (parent_probs * operator_entropy).sum(dim=-1)


__all__ = ["CordControllerOutput", "CordStateController"]
