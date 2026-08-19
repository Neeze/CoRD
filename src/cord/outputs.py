"""Model and search output types for CoRD."""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput


@dataclass
class CordConceptPacket(ModelOutput):
    """Fixed-size concept representation and lightweight archive metadata."""

    states: Optional[torch.FloatTensor] = None
    confidence: Optional[torch.FloatTensor] = None
    provenance: Optional[torch.FloatTensor] = None
    fingerprint: Optional[tuple[str, ...]] = None


@dataclass
class CordModelOutput(ModelOutput):
    """Base-model outputs with recurrent and concept diagnostics."""

    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[object] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    concept_states: Optional[torch.FloatTensor] = None
    concept_confidence: Optional[torch.FloatTensor] = None
    concept_provenance: Optional[torch.FloatTensor] = None
    loop_values: Optional[torch.FloatTensor] = None
    loop_uncertainties: Optional[torch.FloatTensor] = None
    halting_probs: Optional[torch.FloatTensor] = None
    active_loops: Optional[torch.LongTensor] = None
    router_logits: Optional[tuple[torch.FloatTensor, ...]] = None
    router_counts: Optional[tuple[torch.LongTensor, ...]] = None
    decoder_positions: Optional[torch.LongTensor] = None
    decoder_mask: Optional[torch.BoolTensor] = None


@dataclass
class CordCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Causal-LM output with CoRD recurrent diagnostics."""

    concept_states: Optional[torch.FloatTensor] = None
    concept_confidence: Optional[torch.FloatTensor] = None
    concept_provenance: Optional[torch.FloatTensor] = None
    loop_values: Optional[torch.FloatTensor] = None
    loop_uncertainties: Optional[torch.FloatTensor] = None
    halting_probs: Optional[torch.FloatTensor] = None
    active_loops: Optional[torch.LongTensor] = None
    router_logits: Optional[tuple[torch.FloatTensor, ...]] = None
    router_counts: Optional[tuple[torch.LongTensor, ...]] = None
    loss_breakdown: Optional[dict[str, torch.Tensor]] = None


@dataclass
class CordSearchOutput(ModelOutput):
    """Result of reward-guided latent state search."""

    best_state: Optional[torch.FloatTensor] = None
    best_state_id: Optional[str] = None
    state_ids: Optional[tuple[str, ...]] = None
    parent_ids: Optional[tuple[str, ...]] = None
    operators: Optional[tuple[str, ...]] = None
    values: Optional[torch.FloatTensor] = None
    rewards: Optional[torch.FloatTensor] = None
    uncertainties: Optional[torch.FloatTensor] = None
    novelties: Optional[torch.FloatTensor] = None
    verified: Optional[tuple[bool, ...]] = None
    decoded_tokens: Optional[tuple[tuple[int, ...], ...]] = None
    verifier_valid: Optional[tuple[Optional[bool], ...]] = None
    verifier_reasons: Optional[tuple[str, ...]] = None
    resource_costs: Optional[torch.FloatTensor] = None
    trajectory: Optional[tuple[dict, ...]] = None
    num_expansions: Optional[int] = None
    selected_state_id: Optional[str] = None
    selected_index: Optional[int] = None
    selected_reward: Optional[float] = None
    oracle_best_reward: Optional[float] = None
    returns: Optional[torch.FloatTensor] = None
    advantages: Optional[torch.FloatTensor] = None
    policy_log_probs: Optional[torch.FloatTensor] = None
    action_parent_ids: Optional[tuple[Optional[str], ...]] = None
    second_parent_ids: Optional[tuple[Optional[str], ...]] = None
    replay_transitions: Optional[tuple[object, ...]] = None


__all__ = [
    "CordConceptPacket",
    "CordModelOutput",
    "CordCausalLMOutputWithPast",
    "CordSearchOutput",
]
