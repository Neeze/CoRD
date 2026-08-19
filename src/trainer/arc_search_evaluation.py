"""Matched-budget reporting for direct ARC decoding and detached graph search."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from cord import CordForCausalLM, CordSearchConfig
from dataset.data import IGNORE_INDEX, PAD_TOKEN

from .arc_verifier import ARCDecodedVerifier


@dataclass
class ARCSearchComparison:
    """Aggregate direct-vs-graph outcomes without conflating their budgets."""

    examples: int = 0
    direct_valid: int = 0
    direct_exact: int = 0
    graph_selected_valid: int = 0
    graph_selected_exact: int = 0
    graph_oracle_exact: int = 0
    graph_expansions: int = 0
    graph_decoded_leaves: int = 0
    graph_decoded_tokens: int = 0
    graph_resource_cost: float = 0.0
    direct_latency_s: float = 0.0
    graph_latency_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        if not self.examples:
            raise ValueError("ARC search comparison produced no examples")
        selected_accuracy = self.graph_selected_exact / self.examples
        return {
            "examples": float(self.examples),
            "direct_valid_grid_rate": self.direct_valid / self.examples,
            "direct_grid_exact": self.direct_exact / self.examples,
            "graph_selected_valid_grid_rate": self.graph_selected_valid / self.examples,
            "selected_solution_accuracy": selected_accuracy,
            "oracle_success_at_n": self.graph_oracle_exact / self.examples,
            "average_graph_expansions": self.graph_expansions / self.examples,
            "average_graph_decoded_leaves": self.graph_decoded_leaves / self.examples,
            "average_graph_decoded_tokens": self.graph_decoded_tokens / self.examples,
            "average_graph_resource_cost": self.graph_resource_cost / self.examples,
            "average_direct_latency_s": self.direct_latency_s / self.examples,
            "average_graph_latency_s": self.graph_latency_s / self.examples,
            # Compatibility keys now use policy-selected quality, never oracle leaf quality.
            "graph_valid_grid_rate": self.graph_selected_valid / self.examples,
            "graph_grid_exact": selected_accuracy,
            "graph_expansions": float(self.graph_expansions),
            "graph_decoded_leaves": float(self.graph_decoded_leaves),
            "graph_resource_cost": self.graph_resource_cost,
        }


@torch.no_grad()
def compare_direct_and_graph(
    model: CordForCausalLM,
    dataloader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    search_config: CordSearchConfig,
    max_steps: int | None = None,
    decode_max_new_tokens: int = 1024,
) -> dict[str, float]:
    """Score direct greedy and graph leaves against identical per-query caps.

    The report exposes graph leaf count/cost rather than implying that a
    multi-leaf search costs the same as one direct greedy decode.  Consumers
    can therefore select matched settings before making a performance claim.
    """
    model.eval()
    if decode_max_new_tokens < 1:
        raise ValueError("decode_max_new_tokens must be positive")
    aggregate = ARCSearchComparison()
    for batch_index, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        prefix_lengths = batch["prefix_lengths"].to(device)
        target_ids = batch["target_ids"].to(device)
        for row, task_id in enumerate(batch["task_ids"]):
            prefix_length = int(prefix_lengths[row].item())
            target_length = int(target_ids[row].ne(IGNORE_INDEX).sum().item())
            if prefix_length < 1 or target_length < 1:
                raise ValueError(f"invalid ARC comparison sample {task_id}")
            prompt = input_ids[row : row + 1, :prefix_length]
            target_grid = batch["target_grids"][row]
            verifier = ARCDecodedVerifier(target_grid)
            direct_start = time.monotonic()
            generated = model.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                prefix_lengths=prompt.new_tensor([prefix_length]),
                max_new_tokens=decode_max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=PAD_TOKEN,
            )
            aggregate.direct_latency_s += time.monotonic() - direct_start
            direct_tokens = generated[0, prompt.shape[1] :].tolist()
            direct_result = verifier(direct_tokens)
            aggregate.direct_valid += int(direct_result.valid)
            aggregate.direct_exact += int(direct_result.exact_reward > 0.0)

            graph_start = time.monotonic()
            search = model.search(
                prompt,
                prefix_lengths=prompt.new_tensor([prefix_length]),
                search_config=search_config,
                verifier=verifier,
                max_new_tokens=decode_max_new_tokens,
                task_id=task_id,
            )
            aggregate.graph_latency_s += time.monotonic() - graph_start
            if search.verifier_valid is None or search.rewards is None:
                raise RuntimeError("graph search did not return terminal verifier diagnostics")
            if search.selected_index is None:
                raise RuntimeError("graph search did not expose its policy-selected leaf")
            selected_index = int(search.selected_index)
            aggregate.graph_selected_valid += int(search.verifier_valid[selected_index] is True)
            aggregate.graph_selected_exact += int(bool(search.verified[selected_index]))
            aggregate.graph_oracle_exact += int(any(search.verified))
            aggregate.graph_expansions += int(search.num_expansions or 0)
            aggregate.graph_decoded_leaves += len(search.decoded_tokens or ())
            aggregate.graph_decoded_tokens += sum(len(tokens) for tokens in (search.decoded_tokens or ()))
            aggregate.graph_resource_cost += float(search.resource_costs.sum().item()) if search.resource_costs is not None else 0.0
            aggregate.examples += 1
        if max_steps is not None and batch_index + 1 >= max_steps:
            break
    return aggregate.as_dict()


__all__ = ["ARCSearchComparison", "compare_direct_and_graph"]
