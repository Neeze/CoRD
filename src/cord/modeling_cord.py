"""Hugging Face model classes for the CoRD architecture."""

from __future__ import annotations

import hashlib
import math
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin

from .cache_utils import CordCache
from .configuration_cord import CordConfig
from .modules import (
    CordBlockAttnRes,
    CordCausalSelfAttention,
    CordCrossAttention,
    CordExpert,
    CordKDA,
    CordLatentMoE,
    CordMLA,
    CordRMSNorm,
    CordSwiGLU,
)
from .outputs import CordCausalLMOutputWithPast, CordConceptPacket, CordModelOutput, CordSearchOutput
from .state_graph import (
    CordOperator,
    CordDecodeContext,
    CordSearchConfig,
    CordStateArchive,
    CordStateRecord,
    CordVerifierResult,
    state_fingerprint,
)
from .training import CordLossTargets, compute_cord_loss


class CordConceptEncoder(nn.Module):
    """Compress a prompt prefix into a fixed number of concept slots."""

    def __init__(self, config: CordConfig):
        super().__init__()
        self.num_slots = config.concept_slots
        self.hidden_size = config.hidden_size
        self.num_heads = config.concept_num_attention_heads
        self.latent_size = config.concept_latent_size
        self.head_dim = self.latent_size // self.num_heads
        self.slot_queries = nn.Parameter(torch.empty(config.concept_slots, config.hidden_size))
        self.query_proj = nn.Linear(config.hidden_size, self.latent_size, bias=False)
        self.key_proj = nn.Linear(config.hidden_size, self.latent_size, bias=False)
        self.value_proj = nn.Linear(config.hidden_size, self.latent_size, bias=False)
        self.output_proj = nn.Linear(self.latent_size, config.hidden_size, bias=False)
        self.norm = CordRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        prompt_embeddings: torch.Tensor,
        prefix_lengths: torch.LongTensor,
        output_provenance: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> CordConceptPacket:
        batch_size, sequence_length, _ = prompt_embeddings.shape
        positions = torch.arange(sequence_length, device=prompt_embeddings.device)
        prompt_mask = positions.unsqueeze(0) < prefix_lengths.unsqueeze(1)
        if attention_mask is not None:
            prompt_mask = prompt_mask & attention_mask.to(torch.bool)
        masked_embeddings = prompt_embeddings * prompt_mask.unsqueeze(-1).to(prompt_embeddings.dtype)
        queries = self.query_proj(self.slot_queries).view(self.num_slots, self.num_heads, self.head_dim)
        queries = queries.transpose(0, 1).unsqueeze(0).expand(batch_size, -1, -1, -1)
        keys = self.key_proj(masked_embeddings).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        values = self.value_proj(masked_embeddings).view(batch_size, sequence_length, self.num_heads, self.head_dim)
        values = values.transpose(1, 2)
        scores = torch.matmul(queries.float(), keys.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        safe_mask = prompt_mask[:, None, None, :]
        no_prompt = ~safe_mask.any(dim=-1, keepdim=True)
        safe_mask = safe_mask | no_prompt
        scores = scores.masked_fill(~safe_mask, torch.finfo(scores.dtype).min)
        weights = scores.softmax(dim=-1).to(prompt_embeddings.dtype)
        attended = torch.matmul(weights, values).transpose(1, 2).reshape(batch_size, self.num_slots, -1)
        has_prompt = prompt_mask.any(dim=-1)
        attended = attended * has_prompt[:, None, None].to(attended.dtype)
        states = self.norm(self.slot_queries.unsqueeze(0) + self.output_proj(attended))
        confidence = weights.max(dim=-1).values.mean(dim=1)
        confidence = confidence * has_prompt[:, None].to(confidence.dtype)
        return CordConceptPacket(
            states=states,
            confidence=confidence,
            provenance=weights.mean(dim=1) if output_provenance else None,
        )

class CordRecurrentMacroBlock(nn.Module):
    """Shared 3-KDA plus 1-MLA recurrent core."""

    def __init__(self, config: CordConfig):
        super().__init__()
        self.config = config
        self.kda_layers = nn.ModuleList([CordKDA(config) for _ in range(config.num_kda_layers)])
        self.mla_layers = nn.ModuleList([CordMLA(config) for _ in range(config.num_mla_layers)])
        attention_count = config.num_kda_layers + config.num_mla_layers
        self.attention_norms = nn.ModuleList(
            [CordRMSNorm(config.hidden_size, config.rms_norm_eps) for _ in range(attention_count)]
        )
        self.expert_bank = nn.ModuleList(
            [
                CordExpert(
                    config.hidden_size,
                    config.routed_latent_width,
                    config.expert_intermediate_size,
                    config.rms_norm_eps,
                    config.use_situ_glu,
                )
                for _ in range(config.num_experts)
            ]
        )
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
        self.moe_layers = nn.ModuleList(
            [
                CordLatentMoE(config, experts=self.expert_bank, shared_expert=self.shared_expert)
                for _ in range(attention_count)
            ]
        ) if config.use_moe else nn.ModuleList()
        self.dense_fallback = CordSwiGLU(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            config.use_situ_glu,
        )
        self.dense_fallback_norm = CordRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.block_attn_res = (
            CordBlockAttnRes(config.hidden_size, config.block_attn_res_slots, config.rms_norm_eps)
            if config.use_block_attn_res
            else None
        )
        self.dropout = nn.Dropout(config.loop_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_states: Optional[list[torch.Tensor]] = None,
        summaries: Optional[list[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor],
    ]:
        recurrent_states = recurrent_states or [None] * len(self.kda_layers)
        summaries = summaries or []
        new_recurrent_states: list[torch.Tensor] = []
        attentions: list[torch.Tensor] = []
        router_logits: list[torch.Tensor] = []
        router_counts: list[torch.Tensor] = []
        attention_index = 0
        for layer_index, layer in enumerate(self.kda_layers):
            residual = hidden_states
            attention_output, recurrent_state = layer(
                self.attention_norms[attention_index](hidden_states), recurrent_states[layer_index]
            )
            hidden_states = residual + self.dropout(attention_output)
            new_recurrent_states.append(recurrent_state)
            if self.moe_layers:
                moe_output, logits, counts = self.moe_layers[attention_index](hidden_states)
                hidden_states = hidden_states + self.dropout(moe_output)
                router_logits.append(logits)
                router_counts.append(counts)
            attention_index += 1
        for layer in self.mla_layers:
            residual = hidden_states
            attention_output, layer_attention = layer(
                self.attention_norms[attention_index](hidden_states), output_attentions
            )
            hidden_states = residual + self.dropout(attention_output)
            if layer_attention is not None:
                attentions.append(layer_attention)
            if self.moe_layers:
                moe_output, logits, counts = self.moe_layers[attention_index](hidden_states)
                hidden_states = hidden_states + self.dropout(moe_output)
                router_logits.append(logits)
                router_counts.append(counts)
            attention_index += 1
        if not self.moe_layers:
            hidden_states = hidden_states + self.dropout(self.dense_fallback(self.dense_fallback_norm(hidden_states)))
        if self.block_attn_res is not None:
            hidden_states, summaries = self.block_attn_res(hidden_states, summaries)
        return hidden_states, new_recurrent_states, summaries, attentions, router_logits, router_counts


class CordDecoderLayer(nn.Module):
    def __init__(self, config: CordConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = CordRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CordCausalSelfAttention(config)
        self.post_attention_layernorm = CordRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.cross_attn = CordCrossAttention(config)
        self.post_cross_attention_layernorm = CordRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = CordSwiGLU(
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            config.use_situ_glu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        concept_states: torch.Tensor,
        concept_mask: Optional[torch.Tensor],
        cache: Optional[CordCache],
        position_ids: Optional[torch.LongTensor],
        output_attentions: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        self_attention, attention_weights = self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask,
            cache,
            self.layer_idx,
            position_ids,
            output_attentions,
        )
        hidden_states = residual + self_attention
        hidden_states = hidden_states + self.cross_attn(
            self.post_attention_layernorm(hidden_states), concept_states, concept_mask
        )
        hidden_states = hidden_states + self.mlp(self.post_cross_attention_layernorm(hidden_states))
        return hidden_states, attention_weights


class CordPreTrainedModel(PreTrainedModel):
    config_class = CordConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CordDecoderLayer", "CordRecurrentMacroBlock"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0.0, std=std)


class CordModel(CordPreTrainedModel):
    """Base CoRD model returning completion hidden states and concept packets."""

    def __init__(self, config: CordConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.concept_encoder = CordConceptEncoder(config)
        self.recurrent_core = CordRecurrentMacroBlock(config)
        self.decoder_layers = nn.ModuleList(
            [CordDecoderLayer(config, layer_idx) for layer_idx in range(config.num_decoder_layers)]
        )
        self.decoder_norm = CordRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.value_head = nn.Linear(config.hidden_size, 1)
        self.uncertainty_head = nn.Linear(config.hidden_size, 1)
        self.halting_head = nn.Linear(config.hidden_size, 1)
        self.operator_embeddings = nn.Parameter(torch.empty(config.num_reasoning_operators, config.hidden_size))
        self.rollback_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.merge_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.merge_cross_attention = CordCrossAttention(config)
        self.post_init()
        nn.init.normal_(self.concept_encoder.slot_queries, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.operator_embeddings, mean=0.0, std=config.initializer_range)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def encode_prompt(
        self,
        input_ids: torch.LongTensor,
        prefix_lengths: torch.LongTensor,
        output_provenance: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> CordConceptPacket:
        packet = self.concept_encoder(
            self.embed_tokens(input_ids),
            prefix_lengths,
            output_provenance=output_provenance,
            attention_mask=attention_mask,
        )
        packet.fingerprint = tuple(state_fingerprint(state) for state in packet.states)
        return packet

    def _resolve_prefix_lengths(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor],
        prefix_lengths: Optional[torch.LongTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.LongTensor:
        if prefix_lengths is not None:
            resolved = prefix_lengths.to(device=input_ids.device, dtype=torch.long)
        elif labels is not None:
            resolved_rows = []
            for row_index, label_row in enumerate(labels):
                supervised = (label_row != -100).nonzero(as_tuple=False)
                if supervised.numel():
                    resolved_rows.append(int(supervised[0].item()))
                elif attention_mask is not None:
                    resolved_rows.append(int(attention_mask[row_index].sum().item()))
                else:
                    resolved_rows.append(input_ids.shape[1])
            resolved = torch.tensor(resolved_rows, device=input_ids.device, dtype=torch.long)
        elif attention_mask is not None:
            resolved = attention_mask.to(device=input_ids.device, dtype=torch.long).sum(dim=-1)
        else:
            resolved = input_ids.new_full((input_ids.shape[0],), input_ids.shape[1])
        if resolved.ndim != 1 or resolved.shape[0] != input_ids.shape[0]:
            raise ValueError("prefix_lengths must have one value per input batch row")
        if (resolved < 0).any() or (resolved > input_ids.shape[1]).any():
            raise ValueError("prefix_lengths must lie between 0 and the input sequence length")
        return resolved

    def _completion_inputs(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: torch.Tensor,
        prefix_lengths: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, torch.LongTensor, torch.BoolTensor]:
        batch_size, sequence_length = input_ids.shape
        positions = prefix_lengths.unsqueeze(1) + torch.arange(
            sequence_length, device=input_ids.device
        ).unsqueeze(0)
        safe_positions = positions.clamp_max(sequence_length - 1)
        valid = positions < sequence_length
        if attention_mask is not None:
            valid = valid & attention_mask.to(torch.bool).gather(1, safe_positions)
        if labels is not None:
            supervised = labels.to(device=input_ids.device).ne(-100).gather(1, safe_positions)
            valid = valid & supervised
        completion_lengths = valid.sum(dim=-1)
        max_completion = int(completion_lengths.max().item()) if completion_lengths.numel() else 0
        if max_completion == 0:
            bos = input_ids.new_full((batch_size, 1), self.config.bos_token_id)
            return (
                self.embed_tokens(bos),
                bos.new_zeros((batch_size, 1)),
                bos.new_ones((batch_size, 1), dtype=torch.bool),
            )
        positions = positions[:, :max_completion]
        valid = valid[:, :max_completion]
        safe_positions = safe_positions[:, :max_completion]
        if labels is None:
            decoder_embeddings = input_embeddings.gather(
                1, safe_positions.unsqueeze(-1).expand(-1, -1, input_embeddings.shape[-1])
            ) * valid.unsqueeze(-1).to(input_embeddings.dtype)
        else:
            decoder_embeddings = input_embeddings.new_zeros(
                batch_size,
                max_completion,
                input_embeddings.shape[-1],
            )
            bos = input_ids.new_full((batch_size,), self.config.bos_token_id)
            decoder_embeddings[:, 0] = self.embed_tokens(bos)
            if max_completion > 1:
                previous_positions = safe_positions[:, :-1]
                previous_embeddings = input_embeddings.gather(
                    1,
                    previous_positions.unsqueeze(-1).expand(-1, -1, input_embeddings.shape[-1]),
                )
                decoder_embeddings[:, 1:] = previous_embeddings * valid[:, :-1].unsqueeze(-1).to(
                    input_embeddings.dtype
                )
        no_completion = ~valid.any(dim=-1)
        if no_completion.any():
            bos = input_ids.new_full((int(no_completion.sum().item()),), self.config.bos_token_id)
            decoder_embeddings[no_completion, 0] = self.embed_tokens(bos)
            safe_positions[no_completion, 0] = prefix_lengths[no_completion].clamp_max(sequence_length - 1)
            valid[no_completion, 0] = True
        return decoder_embeddings, safe_positions, valid

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[CordCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        prefix_lengths: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_provenance: bool = False,
        return_dict: Optional[bool] = None,
        num_recurrent_loops: Optional[int] = None,
        **kwargs,
    ) -> tuple | CordModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("exactly one of input_ids or inputs_embeds must be specified")
        if inputs_embeds is not None:
            if input_ids is None:
                input_ids = inputs_embeds.new_zeros(inputs_embeds.shape[:-1], dtype=torch.long)
            input_embeddings = inputs_embeds
        else:
            input_embeddings = self.embed_tokens(input_ids)
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != input_ids.shape[0]:
                raise ValueError("attention_mask must be a two-dimensional mask with one row per input")
            if past_key_values is None and attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape during the initial forward pass")
            if past_key_values is not None and attention_mask.shape[1] < input_ids.shape[1]:
                raise ValueError("cached attention_mask must include the current input tokens")
        if labels is not None and labels.shape != input_ids.shape:
            raise ValueError("labels must match input_ids shape for prefix-causal training")
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        use_cache = self.config.use_cache if use_cache is None else use_cache
        if past_key_values is not None and not isinstance(past_key_values, CordCache):
            raise TypeError("CoRD requires a CordCache for cached decoding")
        if past_key_values is None:
            resolved_prefix_lengths = self._resolve_prefix_lengths(
                input_ids,
                labels,
                prefix_lengths,
                attention_mask,
            )
            concept_packet = self.concept_encoder(
                input_embeddings,
                resolved_prefix_lengths,
                output_provenance=output_provenance,
                attention_mask=attention_mask,
            )
            loops = self.config.num_recurrent_loops if num_recurrent_loops is None else num_recurrent_loops
            if not 1 <= loops <= self.config.max_recurrent_loops:
                raise ValueError("num_recurrent_loops is outside the configured range")
            concept_states = concept_packet.states
            recurrent_states = None
            summaries: list[torch.Tensor] = []
            loop_values = []
            loop_uncertainties = []
            halting_probs = []
            all_router_logits = []
            all_router_counts = []
            active = torch.ones(input_embeddings.shape[0], device=input_embeddings.device, dtype=torch.bool)
            for loop_index in range(loops):
                previous = concept_states
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    def _custom_forward(c_states, r_states, summs):
                        return self.recurrent_core(c_states, r_states, summs, output_attentions)
                    concept_states, recurrent_states, summaries, _, router_logits, router_counts = torch.utils.checkpoint.checkpoint(
                        _custom_forward,
                        concept_states,
                        recurrent_states,
                        summaries,
                        use_reentrant=False,
                    )
                else:
                    concept_states, recurrent_states, summaries, _, router_logits, router_counts = self.recurrent_core(
                        concept_states,
                        recurrent_states,
                        summaries,
                        output_attentions,
                    )
                all_router_logits.extend(router_logits)
                all_router_counts.extend(router_counts)
                concept_states = torch.where(active[:, None, None], concept_states, previous)
                pooled = concept_states.mean(dim=1)
                loop_values.append(self.value_head(pooled).squeeze(-1))
                loop_uncertainties.append(F.softplus(self.uncertainty_head(pooled).squeeze(-1)))
                current_halt = torch.sigmoid(self.halting_head(pooled).squeeze(-1))
                halting_probs.append(current_halt)
                if loop_index + 1 >= self.config.minimum_recurrent_loops:
                    active = active & (current_halt < self.config.halting_threshold)
            halt_mask = torch.stack(halting_probs, dim=1).ge(self.config.halting_threshold)
            halt_mask[:, : self.config.minimum_recurrent_loops - 1] = False
            first_halt = halt_mask.float().argmax(dim=1)
            active_loop_count = torch.where(
                halt_mask.any(dim=1),
                first_halt + 1,
                torch.full_like(first_halt, loops),
            )
            decoder_embeddings, decoder_positions, decoder_mask = self._completion_inputs(
                input_ids,
                input_embeddings,
                resolved_prefix_lengths,
                attention_mask,
                labels,
            )
            cache = CordCache(
                len(self.decoder_layers),
                concept_states=concept_states if use_cache else None,
                concept_confidence=concept_packet.confidence if use_cache else None,
                prefix_lengths=resolved_prefix_lengths if use_cache else None,
                decoder_mask=decoder_mask if use_cache else None,
            ) if use_cache else None
            concept_confidence = concept_packet.confidence
            concept_provenance = concept_packet.provenance
        else:
            cache = past_key_values
            if cache.concept_states is None:
                raise ValueError("CordCache is missing concept states")
            concept_states = cache.concept_states
            concept_confidence = cache.concept_confidence
            concept_provenance = None
            decoder_embeddings = input_embeddings
            decoder_positions = input_ids.new_full((input_ids.shape[0], input_ids.shape[1]), 0)
            current_decoder_mask = (
                attention_mask[:, -input_ids.shape[1] :].to(torch.bool)
                if attention_mask is not None
                else input_ids.new_ones(input_ids.shape, dtype=torch.bool)
            )
            if cache.decoder_mask is None:
                past_length = cache.get_seq_length()
                decoder_mask = torch.cat(
                    (
                        current_decoder_mask.new_ones((input_ids.shape[0], past_length)),
                        current_decoder_mask,
                    ),
                    dim=-1,
                )
            else:
                decoder_mask = torch.cat((cache.decoder_mask, current_decoder_mask), dim=-1)
            cache.decoder_mask = decoder_mask
            loops = 0
            loop_values = []
            loop_uncertainties = []
            halting_probs = []
            all_router_logits = []
            all_router_counts = []
            active_loop_count = input_ids.new_zeros((input_ids.shape[0],))
            output_attentions = bool(output_attentions)
            output_hidden_states = bool(output_hidden_states)

        hidden_states = decoder_embeddings
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        concept_mask = concept_confidence > 0 if concept_confidence is not None else None
        for decoder_layer in self.decoder_layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states, attention_weights = decoder_layer(
                hidden_states,
                decoder_mask,
                concept_states,
                concept_mask,
                cache,
                position_ids,
                output_attentions,
            )
            if output_attentions and attention_weights is not None:
                all_attentions.append(attention_weights)
        hidden_states = self.decoder_norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        loop_values_tensor = torch.stack(loop_values, dim=1) if loop_values else None
        loop_uncertainties_tensor = torch.stack(loop_uncertainties, dim=1) if loop_uncertainties else None
        halting_tensor = torch.stack(halting_probs, dim=1) if halting_probs else None
        if not return_dict:
            return tuple(
                value
                for value in (
                    hidden_states,
                    cache,
                    tuple(all_hidden_states) if all_hidden_states is not None else None,
                    tuple(all_attentions) if all_attentions is not None else None,
                    concept_states,
                    concept_confidence,
                    concept_provenance,
                    loop_values_tensor,
                    loop_uncertainties_tensor,
                    halting_tensor,
                    active_loop_count,
                    decoder_positions,
                    decoder_mask,
                )
                if value is not None
            )
        return CordModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=cache,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
            concept_states=concept_states,
            concept_confidence=concept_confidence,
            concept_provenance=concept_provenance,
            loop_values=loop_values_tensor,
            loop_uncertainties=loop_uncertainties_tensor,
            halting_probs=halting_tensor,
            active_loops=active_loop_count,
            router_logits=tuple(all_router_logits),
            router_counts=tuple(all_router_counts),
            decoder_positions=decoder_positions,
            decoder_mask=decoder_mask,
        )

    def _operator_state(
        self,
        state: torch.Tensor,
        operator: CordOperator,
        other_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        operator_index = list(CordOperator).index(operator)
        operator_embedding = self.operator_embeddings[operator_index].view(1, 1, -1)
        if operator is CordOperator.HALT:
            return state
        if operator is CordOperator.ROLLBACK and other_state is not None:
            gate_input = torch.cat((state, other_state), dim=-1)
            state = other_state + torch.sigmoid(self.rollback_gate(gate_input)) * (state - other_state)
        elif operator is CordOperator.MERGE and other_state is not None:
            query_state = state if state.ndim == 3 else state.unsqueeze(0)
            source_state = other_state if other_state.ndim == 3 else other_state.unsqueeze(0)
            cross_update = self.merge_cross_attention(query_state, source_state)
            if state.ndim == 2:
                cross_update = cross_update.squeeze(0)
            gate_input = torch.cat((state, cross_update), dim=-1)
            state = state + torch.sigmoid(self.merge_gate(gate_input)) * cross_update
        state_3d = state if state.ndim == 3 else state.unsqueeze(0)
        transitioned, _, _, _, _, _ = self.recurrent_core(state_3d + operator_embedding)
        return transitioned if state.ndim == 3 else transitioned.squeeze(0)

    @torch.no_grad()
    def search(
        self,
        input_ids: torch.LongTensor,
        prefix_lengths: Optional[torch.LongTensor] = None,
        search_config: Optional[CordSearchConfig] = None,
        decode_context: Optional[CordDecodeContext] = None,
        decode_state: Optional[Callable[[torch.Tensor, CordDecodeContext], list[int]]] = None,
        verifier: Optional[Callable[[list[int], CordDecodeContext], CordVerifierResult]] = None,
    ) -> CordSearchOutput:
        """Run bounded beam/UCB search over detached concept states."""

        if input_ids.shape[0] != 1:
            raise ValueError("Cord v1 search currently accepts one prompt at a time")
        search_config = search_config or CordSearchConfig()
        if prefix_lengths is None:
            prefix_lengths = input_ids.new_tensor([input_ids.shape[1]])
        packet = self.encode_prompt(input_ids, prefix_lengths)
        root_state = packet.states[0].detach()
        archive = CordStateArchive(
            self.config.hot_frontier_max_size,
            self.config.gpu_checkpoint_slots,
            self.config.archive_checkpoint_interval,
        )

        def replay_state(state: torch.Tensor, record: CordStateRecord) -> torch.Tensor:
            other_state = None
            if record.operator in {CordOperator.ROLLBACK, CordOperator.MERGE} and len(record.parent_ids) > 1:
                other_state = archive.materialize(record.parent_ids[1], replay_state, input_ids.device)
            return self._operator_state(state, record.operator, other_state)

        root_id = state_fingerprint(root_state)
        archive.add(
            CordStateRecord(
                root_id,
                (),
                CordOperator.CONTINUE,
                0,
                0.0,
                0.0,
                0.0,
                1.0,
                root_id,
                search_config.seed,
                state=root_state,
            )
        )
        frontier = [root_id]
        expanded_pairs: set[tuple[str, CordOperator]] = set()
        generator = torch.Generator(device=input_ids.device).manual_seed(search_config.seed)
        for expansion_index in range(search_config.max_expansions):
            candidates: list[tuple[float, CordStateRecord, torch.Tensor]] = []
            for parent_id in frontier:
                parent = archive.get(parent_id)
                parent_state = archive.materialize(parent_id, replay_state, input_ids.device)
                for operator in CordOperator:
                    if operator is CordOperator.HALT or (parent_id, operator) in expanded_pairs:
                        continue
                    expanded_pairs.add((parent_id, operator))
                    other_id = None
                    if operator is CordOperator.MERGE and frontier[0] != parent_id:
                        other_id = frontier[0]
                    elif operator is CordOperator.ROLLBACK and parent.parent_ids:
                        other_id = parent.parent_ids[0]
                        # Prefer an older archived ancestor when available: rollback
                        # is a graph operation, not merely an undo of the last edge.
                        while archive.get(other_id).parent_ids:
                            other_id = archive.get(other_id).parent_ids[0]
                    other_state = (
                        archive.materialize(other_id, replay_state, input_ids.device)
                        if other_id is not None
                        else None
                    )
                    state = self._operator_state(parent_state, operator, other_state).detach()
                    fingerprint = state_fingerprint(state)
                    if any(record.fingerprint == fingerprint for record in archive.records.values()):
                        continue
                    pooled = state.mean(dim=0)
                    value = float(self.value_head(pooled).item())
                    uncertainty = float(F.softplus(self.uncertainty_head(pooled)).item())
                    novelty = 1.0
                    random_bonus = (
                        float(torch.rand((), generator=generator, device=state.device).item())
                        if not search_config.deterministic
                        else 0.0
                    )
                    score = value + search_config.exploration_weight * (uncertainty + random_bonus)
                    score += search_config.novelty_weight * novelty
                    score -= search_config.compute_cost_weight * (parent.depth + 1)
                    score -= search_config.memory_cost_weight * float(archive.hot_residency)
                    state_id = hashlib_state_id(parent_id, operator.value, expansion_index, fingerprint)
                    reward = value
                    verified = False
                    record = CordStateRecord(
                        state_id,
                        (parent_id,) if other_state is None else (parent_id, frontier[0]),
                        operator,
                        parent.depth + 1,
                        value,
                        reward,
                        uncertainty,
                        novelty,
                        fingerprint,
                        search_config.seed + expansion_index,
                        verified,
                        provenance_ids=tuple(range(state.shape[0])),
                        state=state,
                    )
                    candidates.append((score, record, state))
            if not candidates:
                break
            candidates.sort(key=lambda candidate: candidate[0], reverse=True)
            frontier = []
            for _, record, _ in candidates[: search_config.beam_size]:
                archive.add(record)
                frontier.append(record.state_id)
        terminal_ids: list[str] = []
        if frontier:
            if verifier is not None and (decode_context is None or decode_state is None):
                raise ValueError("decoded verification requires both decode_context and decode_state")
            for terminal_index, terminal_parent in enumerate(frontier[: search_config.max_verified_leaves]):
                terminal_state = archive.materialize(terminal_parent, replay_state, input_ids.device)
                terminal_id = hashlib_state_id(
                    terminal_parent,
                    CordOperator.HALT.value,
                    len(archive.records),
                    state_fingerprint(terminal_state),
                )
                terminal_record = archive.get(terminal_parent)
                decoded_tokens: tuple[int, ...] = ()
                result = None
                if decode_state is not None and decode_context is not None:
                    decoded_tokens = tuple(int(token) for token in decode_state(terminal_state, decode_context))
                if verifier is not None:
                    result = verifier(list(decoded_tokens), decode_context)
                    if not isinstance(result, CordVerifierResult):
                        raise TypeError("state-graph verifier must return CordVerifierResult")
                terminal_reward = result.reward if result is not None else terminal_record.reward
                terminal_verified = bool(result is not None and result.valid and result.exact_reward > 0.0)
                resource_cost = float(terminal_record.depth + 1 + len(decoded_tokens))
                archive.add(
                    CordStateRecord(
                        terminal_id,
                        (terminal_parent,),
                        CordOperator.HALT,
                        terminal_record.depth + 1,
                        terminal_record.value,
                        terminal_reward,
                        terminal_record.uncertainty,
                        terminal_record.novelty,
                        state_fingerprint(terminal_state),
                        search_config.seed + terminal_index,
                        terminal_verified,
                        provenance_ids=terminal_record.provenance_ids,
                        decoded_tokens=decoded_tokens,
                        verifier_valid=result.valid if result is not None else None,
                        verifier_reason=result.reason if result is not None else "",
                        resource_cost=resource_cost,
                        state=terminal_state,
                    )
                )
                terminal_ids.append(terminal_id)
        if terminal_ids:
            frontier = terminal_ids
        records = [archive.get(state_id) for state_id in frontier]
        best = (
            max(records, key=lambda record: (record.verified, record.reward, record.value))
            if records
            else archive.get(root_id)
        )
        best_state = archive.materialize(best.state_id, replay_state, input_ids.device)
        prompt_identity = hashlib.sha1(input_ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        return CordSearchOutput(
            best_state=best_state,
            best_state_id=best.state_id,
            state_ids=tuple(record.state_id for record in records),
            parent_ids=tuple(record.parent_ids[0] if record.parent_ids else "" for record in records),
            operators=tuple(record.operator.value for record in records),
            values=torch.tensor([record.value for record in records], device=input_ids.device),
            rewards=torch.tensor([record.reward for record in records], device=input_ids.device),
            uncertainties=torch.tensor([record.uncertainty for record in records], device=input_ids.device),
            novelties=torch.tensor([record.novelty for record in records], device=input_ids.device),
            verified=tuple(record.verified for record in records),
            decoded_tokens=tuple(record.decoded_tokens for record in records),
            verifier_valid=tuple(record.verifier_valid for record in records),
            verifier_reasons=tuple(record.verifier_reason for record in records),
            resource_costs=torch.tensor([record.resource_cost for record in records], device=input_ids.device),
            trajectory=tuple(
                {
                    "task_id": decode_context.task_id if decode_context is not None else None,
                    "prompt_identity": prompt_identity,
                    "state_id": record.state_id,
                    "parent_ids": record.parent_ids,
                    "operator": record.operator.value,
                    "depth": record.depth,
                    "value": record.value,
                    "reward": record.reward,
                    "uncertainty": record.uncertainty,
                    "decoded_tokens": record.decoded_tokens,
                    "verifier_valid": record.verifier_valid,
                    "verifier_reason": record.verifier_reason,
                    "rng_seed": record.rng_seed,
                    "resource_cost": record.resource_cost,
                }
                for record in archive.records.values()
            ),
            num_expansions=len(archive.records) - 1,
        )


def hashlib_state_id(parent_id: str, operator: str, expansion_index: int, fingerprint: str) -> str:
    return f"{parent_id[:8]}-{operator}-{expansion_index}-{fingerprint[:8]}"


class CordForCausalLM(CordPreTrainedModel, GenerationMixin):
    """Prefix-causal CoRD language model with a tied output head."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: CordConfig):
        super().__init__(config)
        self.model = CordModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value
        if self.config.tie_word_embeddings:
            self.lm_head.weight = value.weight

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    @torch.no_grad()
    def decode_concept_state(
        self,
        concept_state: torch.Tensor,
        context: CordDecodeContext,
    ) -> list[int]:
        """Greedily decode a detached graph leaf through CoRD's normal cache."""
        if concept_state.ndim != 2:
            raise ValueError("a graph leaf must have shape (concept_slots, hidden_size)")
        if context.input_ids.shape[0] != 1:
            raise ValueError("decoded graph leaves currently require one prompt")
        state = concept_state.unsqueeze(0)
        cache = CordCache(
            len(self.model.decoder_layers),
            concept_states=state,
            concept_confidence=torch.ones(state.shape[:2], dtype=state.dtype, device=state.device),
            prefix_lengths=context.prefix_lengths.to(device=state.device, dtype=torch.long),
        )
        current = torch.full(
            (1, 1), self.config.bos_token_id, dtype=torch.long, device=state.device
        )
        decoded: list[int] = []
        for _ in range(context.max_new_tokens):
            output = self(
                input_ids=current,
                attention_mask=torch.ones_like(current),
                past_key_values=cache,
                prefix_lengths=context.prefix_lengths,
                use_cache=True,
            )
            cache = output.past_key_values
            token = int(output.logits[:, -1].argmax(dim=-1).item())
            decoded.append(token)
            if token == self.config.eos_token_id:
                break
            current = current.new_tensor([[token]])
        return decoded

    def search(
        self,
        input_ids: torch.LongTensor,
        prefix_lengths: Optional[torch.LongTensor] = None,
        search_config: Optional[CordSearchConfig] = None,
        verifier: Optional[Callable[[list[int], CordDecodeContext], CordVerifierResult]] = None,
        *,
        max_new_tokens: int = 32,
        task_id: Optional[str] = None,
    ) -> CordSearchOutput:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if prefix_lengths is None:
            prefix_lengths = input_ids.new_tensor([input_ids.shape[1]])
        context = CordDecodeContext(input_ids, prefix_lengths, max_new_tokens, task_id)
        return self.model.search(
            input_ids,
            prefix_lengths=prefix_lengths,
            search_config=search_config,
            decode_context=context,
            decode_state=self.decode_concept_state,
            verifier=verifier,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[CordCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        prefix_lengths: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        loss_targets: Optional[CordLossTargets] = None,
        **kwargs,
    ) -> tuple | CordCausalLMOutputWithPast:
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            prefix_lengths=prefix_lengths,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        logits = self.lm_head(model_outputs.last_hidden_state)
        selected_labels = None
        if labels is not None:
            if labels.shape[1] == logits.shape[1]:
                selected_labels = labels
            else:
                positions = model_outputs.decoder_positions.clamp_max(labels.shape[1] - 1)
                selected_labels = labels.gather(1, positions)
            selected_labels = selected_labels.masked_fill(~model_outputs.decoder_mask, -100)
        loss, loss_breakdown = compute_cord_loss(
            logits,
            selected_labels,
            loop_values=model_outputs.loop_values,
            halting_probs=model_outputs.halting_probs,
            concept_states=model_outputs.concept_states,
            router_logits=model_outputs.router_logits,
            targets=loss_targets,
        )
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if not return_dict:
            return tuple(
                value
                for value in ((loss,) if loss is not None else ())
                + (
                    logits,
                    model_outputs.past_key_values,
                    model_outputs.hidden_states,
                    model_outputs.attentions,
                )
                if value is not None
            )
        return CordCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            concept_states=model_outputs.concept_states,
            concept_confidence=model_outputs.concept_confidence,
            concept_provenance=model_outputs.concept_provenance,
            loop_values=model_outputs.loop_values,
            loop_uncertainties=model_outputs.loop_uncertainties,
            halting_probs=model_outputs.halting_probs,
            active_loops=model_outputs.active_loops,
            router_logits=model_outputs.router_logits,
            router_counts=model_outputs.router_counts,
            loss_breakdown=loss_breakdown,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[CordCache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        # Transformers may initialize generation with an empty DynamicCache. CoRD
        # owns a specialized cache, so treat that placeholder as no cache and let
        # the first forward pass construct a CordCache.
        if past_key_values is not None and not isinstance(past_key_values, CordCache):
            cache_length = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else None
            if cache_length == 0:
                past_key_values = None
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        elif prefix_lengths is None:
            prefix_lengths = (
                attention_mask.sum(dim=-1)
                if attention_mask is not None
                else input_ids.new_full((input_ids.shape[0],), input_ids.shape[1])
            )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "prefix_lengths": prefix_lengths,
            "use_cache": True,
            **kwargs,
        }

    @staticmethod
    def _reorder_cache(past_key_values: CordCache, beam_idx: torch.LongTensor) -> CordCache:
        return past_key_values.reorder_cache(beam_idx)


__all__ = [
    "CordPreTrainedModel",
    "CordModel",
    "CordForCausalLM",
    "CordConceptPacket",
    "CordSearchOutput",
]
