"""Loss and optimizer-group utilities for CoRD training experiments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .controller import CordStateController
from .state_graph import CordOperator


@dataclass
class CordLossTargets:
    """Optional targets for the auxiliary recurrent-state objectives."""

    value: Optional[torch.Tensor] = None
    route: Optional[torch.Tensor] = None
    halt: Optional[torch.Tensor] = None
    concept: Optional[torch.Tensor] = None
    teacher_logits: Optional[torch.Tensor] = None


@dataclass
class CordGraphTransition:
    """Replayable controller decision and its offline credit target."""

    current_state: torch.Tensor
    archive_states: torch.Tensor
    goal_state: torch.Tensor
    budget_remaining: float
    parent_index: int
    operator_index: int
    second_parent_index: int
    behavior_log_prob: float
    return_value: float
    advantage: float
    parent_state: torch.Tensor
    second_parent_state: Optional[torch.Tensor]
    child_state: torch.Tensor
    terminal: bool = False
    exact_reward: float = 0.0
    local_initial_state: Optional[torch.Tensor] = None
    local_operator_indices: tuple[int, ...] = ()
    local_second_parent_states: tuple[Optional[torch.Tensor], ...] = ()
    local_target_states: tuple[torch.Tensor, ...] = ()
    decoder_target_ids: Optional[torch.Tensor] = None

    def detached_cpu(self) -> "CordGraphTransition":
        """Drop rollout graphs before long-lived replay storage."""

        def copy(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if tensor is None else tensor.detach().to("cpu")

        return CordGraphTransition(
            current_state=copy(self.current_state),
            archive_states=copy(self.archive_states),
            goal_state=copy(self.goal_state),
            budget_remaining=self.budget_remaining,
            parent_index=self.parent_index,
            operator_index=self.operator_index,
            second_parent_index=self.second_parent_index,
            behavior_log_prob=self.behavior_log_prob,
            return_value=self.return_value,
            advantage=self.advantage,
            parent_state=copy(self.parent_state),
            second_parent_state=copy(self.second_parent_state),
            child_state=copy(self.child_state),
            terminal=self.terminal,
            exact_reward=self.exact_reward,
            local_initial_state=copy(self.local_initial_state),
            local_operator_indices=self.local_operator_indices,
            local_second_parent_states=tuple(copy(item) for item in self.local_second_parent_states),
            local_target_states=tuple(copy(item) for item in self.local_target_states),
            decoder_target_ids=copy(self.decoder_target_ids),
        )


class CordGraphReplayBuffer:
    """Bounded off-policy replay with explicit successful/hard-negative retention."""

    def __init__(self, capacity: int = 4096):
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self._items: deque[CordGraphTransition] = deque(maxlen=capacity)

    def extend(self, transitions) -> None:
        for transition in transitions:
            if not isinstance(transition, CordGraphTransition):
                raise TypeError("graph replay accepts CordGraphTransition values")
            self._items.append(transition.detached_cpu())

    def sample(self, count: int, *, generator: Optional[torch.Generator] = None) -> list[CordGraphTransition]:
        if count < 1:
            raise ValueError("sample count must be positive")
        if not self._items:
            return []
        weights = torch.tensor(
            [2.0 if item.exact_reward > 0.0 else 1.0 + min(abs(item.advantage), 1.0) for item in self._items]
        )
        indices = torch.multinomial(weights, min(count, len(self._items)), replacement=False, generator=generator)
        return [self._items[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self._items)

    def state_dict(self) -> dict:
        return {"capacity": self.capacity, "items": list(self._items)}

    def load_state_dict(self, state: dict) -> None:
        capacity = int(state["capacity"])
        if capacity < 1:
            raise ValueError("serialized replay capacity must be positive")
        items = state.get("items", [])
        if any(not isinstance(item, CordGraphTransition) for item in items):
            raise TypeError("serialized graph replay contains an invalid transition")
        self.capacity = capacity
        self._items = deque(items[-capacity:], maxlen=capacity)


def compute_graph_policy_loss(
    controller: CordStateController,
    transitions: list[CordGraphTransition] | tuple[CordGraphTransition, ...],
    *,
    device: Optional[torch.device] = None,
    awr_temperature: float = 0.5,
    max_advantage_weight: float = 20.0,
    value_weight: float = 0.5,
    uncertainty_weight: float = 0.05,
    entropy_weight: float = 0.01,
    ppo_clip: Optional[float] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """AWR by default, or clipped PPO when ``ppo_clip`` is supplied."""

    if not transitions:
        raise ValueError("graph policy loss requires at least one transition")
    if awr_temperature <= 0.0 or max_advantage_weight <= 0.0:
        raise ValueError("AWR temperature and maximum weight must be positive")
    if ppo_clip is not None and ppo_clip <= 0.0:
        raise ValueError("ppo_clip must be positive")
    if device is None:
        device = next(controller.parameters()).device
    policy_terms = []
    value_terms = []
    uncertainty_terms = []
    entropies = []
    weights = []
    for transition in transitions:
        output = controller(
            transition.current_state.to(device),
            transition.archive_states.to(device),
            transition.goal_state.to(device),
            transition.budget_remaining,
        )
        parent_index = torch.tensor([transition.parent_index], device=device, dtype=torch.long)
        operator_index = torch.tensor([transition.operator_index], device=device, dtype=torch.long)
        second_parent_index = torch.tensor(
            [max(transition.second_parent_index, 0)], device=device, dtype=torch.long
        )
        log_prob = controller.action_log_prob(output, parent_index, operator_index, second_parent_index)
        advantage = log_prob.new_tensor(transition.advantage)
        if ppo_clip is None:
            weight = torch.exp(advantage / awr_temperature).clamp(max=max_advantage_weight).detach()
            policy_terms.append(-weight * log_prob.squeeze(0))
        else:
            ratio = torch.exp((log_prob.squeeze(0) - transition.behavior_log_prob).clamp(-20.0, 20.0))
            clipped = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip)
            policy_terms.append(-torch.minimum(ratio * advantage, clipped * advantage))
            weight = ratio.detach()
        predicted_value = output.values[0, transition.parent_index]
        target = predicted_value.new_tensor(transition.return_value)
        error = target - predicted_value
        sigma = output.uncertainties[0, transition.parent_index].clamp_min(1e-5)
        value_terms.append(F.smooth_l1_loss(predicted_value, target))
        uncertainty_terms.append(0.5 * error.detach().square() / sigma.square() + sigma.log())
        entropies.append(controller.entropy(output).squeeze(0))
        weights.append(weight)
    raw_policy = torch.stack(policy_terms).mean()
    raw_value = torch.stack(value_terms).mean()
    raw_uncertainty = torch.stack(uncertainty_terms).mean()
    entropy = torch.stack(entropies).mean()
    total = raw_policy + value_weight * raw_value + uncertainty_weight * raw_uncertainty - entropy_weight * entropy
    return total, {
        "graph_policy": raw_policy,
        "graph_value": raw_value * value_weight,
        "graph_uncertainty": raw_uncertainty * uncertainty_weight,
        "graph_entropy": entropy,
        "graph_advantage_weight": torch.stack(weights).float().mean(),
    }


def compute_local_transition_loss(
    model: nn.Module,
    transitions: list[CordGraphTransition] | tuple[CordGraphTransition, ...],
    *,
    awr_temperature: float = 0.5,
    max_advantage_weight: float = 20.0,
) -> torch.Tensor:
    """Advantage-weighted local BPTT against replayed latent children.

    Replayed child packets are stale targets from search.  Recomputing only one
    structural transition bounds the recurrent graph while still updating the
    shared transition core, rollback gate, merge attention, and operator code.
    """

    core = model.model if hasattr(model, "model") and hasattr(model.model, "_operator_state") else model
    device = next(core.parameters()).device
    terms = []
    for transition in transitions:
        if transition.local_operator_indices:
            current = transition.local_initial_state.to(device)
            step_terms = []
            for operator_index, other_state, target_state in zip(
                transition.local_operator_indices,
                transition.local_second_parent_states,
                transition.local_target_states,
                strict=True,
            ):
                operator = list(CordOperator)[operator_index]
                if operator is CordOperator.HALT:
                    continue
                other = None if other_state is None else other_state.to(device)
                current = core._operator_state(current, operator, other)
                target = target_state.to(device).detach()
                step_terms.append(
                    F.smooth_l1_loss(current, target)
                    + 1.0 - F.cosine_similarity(current.flatten(), target.flatten(), dim=0)
                )
            if not step_terms:
                continue
            transition_loss = torch.stack(step_terms).mean()
        else:
            operator = list(CordOperator)[transition.operator_index]
            if operator is CordOperator.HALT or transition.terminal:
                continue
            parent = transition.parent_state.to(device)
            other = None if transition.second_parent_state is None else transition.second_parent_state.to(device)
            predicted = core._operator_state(parent, operator, other)
            target = transition.child_state.to(device).detach()
            transition_loss = F.smooth_l1_loss(predicted, target) + 1.0 - F.cosine_similarity(
                predicted.flatten(), target.flatten(), dim=0
            )
        weight = min(torch.exp(torch.tensor(transition.advantage / awr_temperature)).item(), max_advantage_weight)
        terms.append(weight * transition_loss)
    if not terms:
        return next(core.parameters()).new_zeros(())
    return torch.stack(terms).mean()


def _replay_local_leaf(core: nn.Module, transition: CordGraphTransition, device: torch.device) -> torch.Tensor:
    if transition.local_operator_indices:
        current = transition.local_initial_state.to(device)
        for operator_index, other_state in zip(
            transition.local_operator_indices, transition.local_second_parent_states, strict=True
        ):
            operator = list(CordOperator)[operator_index]
            if operator is CordOperator.HALT:
                continue
            other = None if other_state is None else other_state.to(device)
            current = core._operator_state(current, operator, other)
        return current
    operator = list(CordOperator)[transition.operator_index]
    if operator is CordOperator.HALT:
        return transition.child_state.to(device)
    other = None if transition.second_parent_state is None else transition.second_parent_state.to(device)
    return core._operator_state(transition.parent_state.to(device), operator, other)


def compute_graph_decoder_loss(
    model: nn.Module,
    transitions: list[CordGraphTransition] | tuple[CordGraphTransition, ...],
    *,
    awr_temperature: float = 0.5,
    max_advantage_weight: float = 20.0,
) -> torch.Tensor:
    """Ground successful/positive-return local trajectories with token CE."""

    if not hasattr(model, "concept_state_loss"):
        raise ValueError("graph decoder loss requires CordForCausalLM.concept_state_loss")
    core = model.model
    device = next(model.parameters()).device
    terms = []
    for transition in transitions:
        if transition.decoder_target_ids is None or transition.return_value <= 0.0:
            continue
        leaf = _replay_local_leaf(core, transition, device)
        token_loss = model.concept_state_loss(leaf, transition.decoder_target_ids.to(device))
        weight = min(torch.exp(torch.tensor(transition.advantage / awr_temperature)).item(), max_advantage_weight)
        terms.append(weight * token_loss)
    if not terms:
        return next(model.parameters()).new_zeros(())
    return torch.stack(terms).mean()


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


def build_graph_optimizer_param_groups(
    model: nn.Module,
    controller_learning_rate: float,
    *,
    backbone_lr_scale: float = 0.1,
    weight_decay: float = 0.1,
) -> list[dict]:
    """Use a 5--10x lower LR for backbone/local-BPTT than the controller."""

    if controller_learning_rate <= 0.0:
        raise ValueError("controller_learning_rate must be positive")
    if not 0.0 < backbone_lr_scale <= 1.0:
        raise ValueError("backbone_lr_scale must be in (0, 1]")
    controller = getattr(getattr(model, "model", model), "state_controller", None)
    if controller is None:
        raise ValueError("model does not expose a CordStateController")
    controller_ids = {id(parameter) for parameter in controller.parameters()}
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {
        (True, True): [], (True, False): [], (False, True): [], (False, False): []
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_controller = id(parameter) in controller_ids
        use_decay = parameter.ndim >= 2 and "embed" not in name and "lm_head" not in name
        groups[(is_controller, use_decay)].append(parameter)
    result = []
    for (is_controller, use_decay), parameters in groups.items():
        if not parameters:
            continue
        result.append({
            "params": parameters,
            "lr": controller_learning_rate if is_controller else controller_learning_rate * backbone_lr_scale,
            "weight_decay": weight_decay if use_decay else 0.0,
            "component": "controller" if is_controller else "backbone",
        })
    return result


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


__all__ = [
    "CordLossTargets",
    "CordGraphTransition",
    "CordGraphReplayBuffer",
    "compute_cord_loss",
    "compute_graph_policy_loss",
    "compute_local_transition_loss",
    "compute_graph_decoder_loss",
    "build_cord_optimizer_param_groups",
    "build_graph_optimizer_param_groups",
    "update_router_biases",
]
