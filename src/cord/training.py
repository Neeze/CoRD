"""Loss and optimizer-group utilities for CoRD training experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CordLossTargets:
    """Optional targets for the auxiliary recurrent-state objectives."""

    value: Optional[torch.Tensor] = None
    route: Optional[torch.Tensor] = None
    halt: Optional[torch.Tensor] = None
    concept: Optional[torch.Tensor] = None
    teacher_logits: Optional[torch.Tensor] = None


def compute_cord_loss(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    *,
    loop_values: Optional[torch.Tensor] = None,
    halting_probs: Optional[torch.Tensor] = None,
    concept_states: Optional[torch.Tensor] = None,
    router_logits: Optional[tuple[torch.Tensor, ...]] = None,
    targets: Optional[CordLossTargets] = None,
    ignore_index: int = -100,
    concept_weight: float = 0.1,
    value_weight: float = 0.1,
    halt_weight: float = 0.1,
    route_weight: float = 0.1,
    distillation_weight: float = 0.0,
) -> tuple[Optional[torch.Tensor], dict[str, torch.Tensor]]:
    """Compute token and explicitly requested auxiliary losses."""

    targets = targets or CordLossTargets()
    losses: dict[str, torch.Tensor] = {}
    if labels is not None:
        valid = labels.ne(ignore_index)
        if valid.any():
            losses["token"] = F.cross_entropy(logits[valid], labels[valid])
    if loop_values is not None and targets.value is not None:
        losses["value"] = F.smooth_l1_loss(loop_values[..., -1], targets.value) * value_weight
    if halting_probs is not None and targets.halt is not None:
        losses["halt"] = F.binary_cross_entropy(halting_probs, targets.halt.float()) * halt_weight
    if concept_states is not None and targets.concept is not None:
        losses["concept"] = F.mse_loss(concept_states, targets.concept) * concept_weight
    if router_logits and targets.route is not None:
        route_logits = router_logits[-1]
        route_target = targets.route.to(route_logits.device)
        losses["route"] = F.cross_entropy(
            route_logits.reshape(-1, route_logits.shape[-1]), route_target.reshape(-1)
        ) * route_weight
    if targets.teacher_logits is not None and distillation_weight:
        student_log_probs = logits.log_softmax(dim=-1)
        teacher_probs = targets.teacher_logits.softmax(dim=-1)
        losses["distill"] = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * distillation_weight
    total = sum(losses.values()) if losses else None
    return total, losses


def build_cord_optimizer_param_groups(model: nn.Module, weight_decay: float = 0.1) -> list[dict]:
    """Split Muon-compatible matrices from AdamW scalar/special parameters."""

    muon_parameters = []
    adamw_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "e_score_correction_bias" in name or parameter.ndim < 2 or "embed" in name or "lm_head" in name:
            if "e_score_correction_bias" in name or parameter.ndim < 2:
                no_decay_parameters.append(parameter)
            else:
                adamw_parameters.append(parameter)
        else:
            muon_parameters.append(parameter)
    return [
        {"params": muon_parameters, "weight_decay": weight_decay, "optimizer": "muon"},
        {"params": adamw_parameters, "weight_decay": weight_decay, "optimizer": "adamw"},
        {"params": no_decay_parameters, "weight_decay": 0.0, "optimizer": "adamw"},
    ]


@torch.no_grad()
def update_router_biases(
    model: nn.Module,
    token_counts: torch.Tensor | tuple[torch.Tensor, ...],
) -> None:
    """Update all loss-free router biases once per effective optimizer step."""

    router_modules = [
        module for module in model.modules() if hasattr(module, "update_correction_bias")
    ]
    if not router_modules:
        return
    if isinstance(token_counts, tuple):
        if len(token_counts) == 0:
            return
        if len(token_counts) % len(router_modules) != 0:
            raise ValueError("router count entries must align with the shared router modules")
        counts_by_router = [
            sum(token_counts[index::len(router_modules)])
            for index in range(len(router_modules))
        ]
    elif token_counts.ndim == 2 and token_counts.shape[0] == len(router_modules):
        counts_by_router = list(token_counts.unbind(0))
    else:
        counts_by_router = [token_counts] * len(router_modules)

    for module, counts in zip(router_modules, counts_by_router):
        total_tokens = counts.sum().to(dtype=torch.float32)
        if total_tokens.item() <= 0:
            continue
        parameter = next(module.parameters(), None)
        if parameter is None:
            continue
        update = getattr(module, "update_correction_bias")
        update(counts.to(parameter.device), int(total_tokens.item()))


__all__ = ["CordLossTargets", "compute_cord_loss", "build_cord_optimizer_param_groups", "update_router_biases"]
