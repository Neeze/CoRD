"""Reusable neural components for the CoRD architecture."""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .cache_utils import CordCache
from .configuration_cord import CordConfig

try:
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    _FLA_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the optional GPU extra.
    chunk_kda = None
    fused_recurrent_kda = None
    _FLA_AVAILABLE = False


class CordRMSNorm(nn.Module):
    """RMSNorm with an FP32 variance reduction."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (normalized.to(hidden_states.dtype) * self.weight).to(hidden_states.dtype)


class CordRotaryEmbedding(nn.Module):
    def __init__(self, config: CordConfig, head_dim: int):
        super().__init__()
        if head_dim % 2:
            raise ValueError("rotary head dimension must be even")
        inverse_frequency = 1.0 / (
            config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def forward(self, position_ids: torch.LongTensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        frequencies = torch.einsum("bi,j->bij", position_ids.float(), self.inverse_frequency)
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        return frequencies.cos().to(dtype), frequencies.sin().to(dtype)


def apply_rotary(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    first_half, second_half = hidden_states.chunk(2, dim=-1)
    rotated = torch.cat((-second_half, first_half), dim=-1)
    return hidden_states * cos + rotated * sin


class CordKDA(nn.Module):
    """Causal gated linear attention over the concept workspace.

    The eager path implements the reference recurrence with an outer-product
    state. The optional FLA path uses the same projected q/k/v/g/beta tensors
    and returns its recurrent state for the next invocation.
    """

    def __init__(self, config: CordConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        projection_size = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.g_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.out_proj = nn.Linear(projection_size, self.hidden_size, bias=False)
        self.a_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(projection_size))
        self.backend = config.kda_backend
        if self.backend == "fla" and not _FLA_AVAILABLE:
            raise ImportError("kda_backend='fla' requires the optional fla-core package")
        self._warned_eager_fallback = False

    def _project(self, hidden_states: torch.Tensor):
        shape = (*hidden_states.shape[:2], self.num_heads, self.head_dim)
        query = self.q_proj(hidden_states).view(shape)
        key = self.k_proj(hidden_states).view(shape)
        value = self.v_proj(hidden_states).view(shape)
        gate = self.g_proj(hidden_states).view(shape)
        beta = self.beta_proj(hidden_states).float()
        query = F.normalize(query.float(), dim=-1).to(hidden_states.dtype)
        key = F.normalize(key.float(), dim=-1).to(hidden_states.dtype)
        return query, key, value, gate, beta

    def _eager_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, num_heads, head_dim = query.shape
        if recurrent_state is None:
            recurrent_state = query.new_zeros(batch_size, num_heads, head_dim, head_dim)
        outputs = []
        for position in range(sequence_length):
            decay = torch.sigmoid(beta[:, position]).to(query.dtype)
            recurrent_state = decay[:, :, None, None] * recurrent_state
            recurrent_state = recurrent_state + torch.einsum(
                "bhd,bhe->bhde", key[:, position], value[:, position]
            )
            output = torch.einsum("bhd,bhde->bhe", query[:, position], recurrent_state)
            outputs.append(output)
        output = torch.stack(outputs, dim=1) * torch.sigmoid(gate)
        output = output.reshape(batch_size, sequence_length, -1)
        return self.out_proj(output), recurrent_state

    def _fla_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        recurrent_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        operation = fused_recurrent_kda if query.shape[1] == 1 else chunk_kda
        output, recurrent_state = operation(
            q=query,
            k=key,
            v=value,
            g=gate,
            beta=beta,
            A_log=self.a_log,
            dt_bias=self.dt_bias,
            initial_state=recurrent_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            transpose_state_layout=True,
        )
        output = output.reshape(output.shape[0], output.shape[1], -1)
        return self.out_proj(output), recurrent_state

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query, key, value, gate, beta = self._project(hidden_states)
        if self.backend in {"auto", "fla"} and _FLA_AVAILABLE:
            try:
                return self._fla_forward(query, key, value, gate, beta, recurrent_state)
            except (RuntimeError, TypeError, ValueError):
                if self.backend == "fla":
                    raise
                if not self._warned_eager_fallback:
                    warnings.warn("FLA KDA failed; falling back to the eager reference path", RuntimeWarning)
                    self._warned_eager_fallback = True
        return self._eager_forward(query, key, value, gate, beta, recurrent_state)


class CordMLA(nn.Module):
    """Latent key/value attention used in the recurrent concept core."""

    def __init__(self, config: CordConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.latent_kv_size = config.latent_kv_size
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.kv_down_proj = nn.Linear(self.hidden_size, self.latent_kv_size, bias=False)
        self.k_proj = nn.Linear(self.latent_kv_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.latent_kv_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.output_gate = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = CordRotaryEmbedding(config, self.head_dim)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length, _ = hidden_states.shape
        position_ids = torch.arange(sequence_length, device=hidden_states.device).expand(batch_size, -1)
        cos, sin = self.rotary_emb(position_ids, hidden_states.dtype)
        query = self.q_proj(hidden_states).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        latent = self.kv_down_proj(hidden_states)
        key = self.k_proj(latent).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(latent).view(
            batch_size, sequence_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)
        repeat_factor = self.num_heads // self.num_key_value_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = scores.softmax(dim=-1).to(hidden_states.dtype)
        weights = self.attention_dropout(weights)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, sequence_length, -1)
        attended = attended * torch.sigmoid(self.output_gate(hidden_states))
        return self.out_proj(attended), weights if output_attentions else None


def _allowed_attention_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    query_length: int,
    key_length: int,
    past_length: int,
    device: torch.device,
) -> torch.Tensor:
    query_positions = past_length + torch.arange(query_length, device=device)
    key_positions = torch.arange(key_length, device=device)
    allowed = key_positions[None, :] <= query_positions[:, None]
    allowed = allowed.unsqueeze(0).expand(batch_size, -1, -1)
    if attention_mask is not None:
        key_mask = attention_mask[:, -key_length:].to(torch.bool)
        allowed = allowed & key_mask[:, None, :]
    return allowed


class CordCausalSelfAttention(nn.Module):
    """Decoder self-attention with GQA, RoPE and :class:`CordCache`."""

    def __init__(self, config: CordConfig):
        super().__init__()
        self.config = config
        self.layer_head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.layer_head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.layer_head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary_emb = CordRotaryEmbedding(config, self.layer_head_dim)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache: Optional[CordCache],
        layer_idx: int,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, query_length, _ = hidden_states.shape
        past_length = cache.get_seq_length(layer_idx) if cache is not None else 0
        if position_ids is None:
            position_ids = torch.arange(
                past_length, past_length + query_length, device=hidden_states.device
            ).expand(batch_size, -1)
        elif position_ids.shape[-1] != query_length:
            position_ids = position_ids[:, -query_length:]
        cos, sin = self.rotary_emb(position_ids, hidden_states.dtype)
        query = self.q_proj(hidden_states).view(
            batch_size, query_length, self.num_heads, self.layer_head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.layer_head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size, query_length, self.num_key_value_heads, self.layer_head_dim
        ).transpose(1, 2)
        query = apply_rotary(query, cos, sin)
        key = apply_rotary(key, cos, sin)
        if cache is not None:
            key, value = cache.update(key, value, layer_idx)
        repeat_factor = self.num_heads // self.num_key_value_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)
        key_length = key.shape[-2]
        allow = _allowed_attention_mask(
            attention_mask, batch_size, query_length, key_length, past_length, hidden_states.device
        )
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(self.layer_head_dim)
        scores = scores.masked_fill(~allow[:, None], torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).to(hidden_states.dtype)
        weights = self.attention_dropout(weights)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, query_length, -1)
        return self.out_proj(attended), weights if output_attentions else None


class CordCrossAttention(nn.Module):
    def __init__(self, config: CordConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        concept_states: torch.Tensor,
        concept_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        key_length = concept_states.shape[1]
        query = self.query(hidden_states).view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(concept_states).view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(concept_states).view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        if concept_mask is not None:
            concept_mask = concept_mask.to(torch.bool)
            concept_mask = concept_mask | ~concept_mask.any(dim=-1, keepdim=True)
            scores = scores.masked_fill(~concept_mask[:, None, None], torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).to(hidden_states.dtype)
        weights = self.attention_dropout(weights)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, query_length, -1)
        return self.out_proj(attended)


class CordSwiGLU(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "silu",
        use_situ_glu: bool = False,
    ):
        super().__init__()
        if activation != "silu":
            raise ValueError("CoRD currently supports only the SiLU SwiGLU activation")
        self.use_situ_glu = use_situ_glu
        self.situ_beta = 1.0
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if self.use_situ_glu:
            activated_gate = self.situ_beta * torch.tanh(gate / self.situ_beta) * torch.sigmoid(gate)
        else:
            activated_gate = F.silu(gate)
        return self.down_proj(activated_gate * up)


class CordExpert(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        latent_width: int,
        intermediate_size: int,
        eps: float,
        use_situ_glu: bool = False,
    ):
        super().__init__()
        self.use_situ_glu = use_situ_glu
        self.situ_beta = 1.0
        self.down_proj = nn.Linear(hidden_size, latent_width, bias=False)
        self.gate_proj = nn.Linear(latent_width, intermediate_size, bias=False)
        self.up_proj = nn.Linear(latent_width, intermediate_size, bias=False)
        self.inner_down_proj = nn.Linear(intermediate_size, latent_width, bias=False)
        self.output_proj = nn.Linear(latent_width, hidden_size, bias=False)
        self.norm = CordRMSNorm(latent_width, eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        latent = self.norm(self.down_proj(hidden_states))
        gate = self.gate_proj(latent)
        up = self.up_proj(latent)
        if self.use_situ_glu:
            activated_gate = self.situ_beta * torch.tanh(gate / self.situ_beta) * torch.sigmoid(gate)
        else:
            activated_gate = F.silu(gate)
        latent = self.inner_down_proj(activated_gate * up)
        return self.output_proj(latent)


class CordLatentMoE(nn.Module):
    """Sparse latent experts with loss-free router correction bias."""

    def __init__(
        self,
        config: CordConfig,
        experts: Optional[nn.ModuleList] = None,
        shared_expert: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)
        if experts is None:
            self.experts = nn.ModuleList(
                [
                    CordExpert(
                        config.hidden_size,
                        config.routed_latent_width,
                        config.expert_intermediate_size,
                        config.rms_norm_eps,
                        config.use_situ_glu,
                    )
                    for _ in range(self.num_experts)
                ]
            )
        else:
            # The macroblock owns this shared bank; duplicate registration would break safe serialization.
            object.__setattr__(self, "experts", experts)
        if shared_expert is None:
            self.shared_expert = (
                CordExpert(
                    config.hidden_size,
                    config.hidden_size,
                    config.intermediate_size,
                    config.rms_norm_eps,
                    config.use_situ_glu,
                )
                if config.num_shared_experts
                else None
            )
        else:
            # The shared expert is owned by the macroblock for the same reason as the expert bank.
            object.__setattr__(self, "shared_expert", shared_expert)

    def forward(self, hidden_states: torch.Tensor):
        original_shape = hidden_states.shape
        flat_states = hidden_states.reshape(-1, original_shape[-1])
        router_logits = F.linear(flat_states.float(), self.router.weight.float())
        if self.config.router_activation == "sigmoid":
            scores = router_logits.sigmoid()
        else:
            scores = router_logits.softmax(dim=-1)
        selection_scores = scores + self.e_score_correction_bias
        top_indices = selection_scores.topk(self.top_k, dim=-1, sorted=False).indices
        top_weights = scores.gather(dim=-1, index=top_indices)
        if self.config.router_renormalize:
            top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        routed = flat_states.new_zeros(flat_states.shape)
        token_counts = torch.zeros(self.num_experts, device=flat_states.device, dtype=torch.long)
        for expert_index, expert in enumerate(self.experts):
            selected = (top_indices == expert_index).nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue
            token_indices = selected[:, 0]
            choice_indices = selected[:, 1]
            token_counts[expert_index] = token_indices.numel()
            expert_output = expert(flat_states.index_select(0, token_indices))
            routed.index_add_(
                0,
                token_indices,
                expert_output * top_weights[token_indices, choice_indices, None].to(expert_output.dtype),
            )
        if self.shared_expert is not None:
            routed = routed + self.shared_expert(flat_states)
        return routed.view(original_shape), router_logits.view(*original_shape[:-1], self.num_experts), token_counts

    @torch.no_grad()
    def update_correction_bias(self, token_counts: torch.Tensor, total_tokens: Optional[int] = None) -> None:
        if total_tokens is None:
            total_tokens = int(token_counts.sum().item())
        if total_tokens <= 0:
            return
        target = token_counts.float().mean()
        self.e_score_correction_bias.add_(
            self.config.router_bias_update_rate * (target - token_counts.float()) / max(float(total_tokens), 1.0)
        )


class CordBlockAttnRes(nn.Module):
    """Attention residual mixer with a bounded summary history."""

    def __init__(self, hidden_size: int, max_summaries: int, eps: float):
        super().__init__()
        self.max_summaries = max_summaries
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.norm = CordRMSNorm(hidden_size, eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        summaries: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        candidates = summaries + [hidden_states]
        stacked = torch.stack(candidates, dim=1)
        scores = self.score(self.norm(stacked)).squeeze(-1).float()
        weights = scores.softmax(dim=1).to(stacked.dtype)
        mixed = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        new_summaries = (summaries + [mixed])[-self.max_summaries :]
        if len(new_summaries) > self.max_summaries:
            new_summaries = new_summaries[-self.max_summaries :]
        return mixed, new_summaries


__all__ = [
    "CordRMSNorm",
    "CordRotaryEmbedding",
    "CordKDA",
    "CordMLA",
    "CordCausalSelfAttention",
    "CordCrossAttention",
    "CordSwiGLU",
    "CordExpert",
    "CordLatentMoE",
    "CordBlockAttnRes",
]
