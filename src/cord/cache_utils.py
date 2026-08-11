"""Generation cache for the prefix-causal CoRD decoder."""

from __future__ import annotations

from typing import Optional

import torch


class CordCache:
    """Dynamic decoder KV cache plus the immutable prompt concept packet."""

    is_compileable = False

    def __init__(
        self,
        num_layers: int,
        concept_states: Optional[torch.Tensor] = None,
        concept_confidence: Optional[torch.Tensor] = None,
        prefix_lengths: Optional[torch.LongTensor] = None,
        decoder_mask: Optional[torch.BoolTensor] = None,
    ):
        self.key_cache: list[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: list[Optional[torch.Tensor]] = [None] * num_layers
        self.concept_states = concept_states
        self.concept_confidence = concept_confidence
        self.prefix_lengths = prefix_lengths
        self.decoder_mask = decoder_mask

    def __len__(self) -> int:
        return len(self.key_cache)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx is None:
            for cached in self.key_cache:
                if cached is not None:
                    return cached.shape[-2]
            return 0
        cached = self.key_cache[layer_idx]
        return 0 if cached is None else cached.shape[-2]

    def get_mask_sizes(self, cache_position: torch.LongTensor, layer_idx: int) -> tuple[int, int]:
        """Return the dynamic key length and offset expected by Transformers generation."""

        return cache_position.shape[0] + self.get_seq_length(layer_idx), 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
            self.value_cache[layer_idx] = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> "CordCache":
        for layer_idx, (keys, values) in enumerate(zip(self.key_cache, self.value_cache)):
            if keys is not None:
                device_index = beam_idx.to(keys.device)
                self.key_cache[layer_idx] = keys.index_select(0, device_index)
                self.value_cache[layer_idx] = values.index_select(0, device_index)
        if self.concept_states is not None:
            self.concept_states = self.concept_states.index_select(0, beam_idx.to(self.concept_states.device))
        if self.concept_confidence is not None:
            self.concept_confidence = self.concept_confidence.index_select(
                0, beam_idx.to(self.concept_confidence.device)
            )
        if self.prefix_lengths is not None:
            self.prefix_lengths = self.prefix_lengths.index_select(0, beam_idx.to(self.prefix_lengths.device))
        if self.decoder_mask is not None:
            self.decoder_mask = self.decoder_mask.index_select(0, beam_idx.to(self.decoder_mask.device))
        return self

    def __getitem__(self, layer_idx: int) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __iter__(self):
        return iter(zip(self.key_cache, self.value_cache))


__all__ = ["CordCache"]
