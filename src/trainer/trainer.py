"""Reusable supervised training and ARC metrics for CoRD experiments."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from cord import CordGraphReplayBuffer, CordSearchConfig
from cord.training import (
    compute_graph_decoder_loss,
    compute_graph_policy_loss,
    compute_local_transition_loss,
    update_router_biases,
)
from dataset.data import EOS_TOKEN, GRID_OFFSET, IGNORE_INDEX, PAD_TOKEN, ROW_SEP


def decode_arc_grid(tokens: Iterable[int], *, require_eos: bool = True) -> list[list[int]] | None:
    """Decode a generated ARC completion; return None for malformed token streams."""
    rows: list[list[int]] = []
    row: list[int] = []
    saw_eos = False
    for token in tokens:
        token = int(token)
        if token == EOS_TOKEN:
            saw_eos = True
            break
        if GRID_OFFSET <= token < GRID_OFFSET + 10:
            row.append(token - GRID_OFFSET)
        elif token == ROW_SEP:
            if not row:
                return None
            rows.append(row)
            row = []
        else:
            return None
    if not saw_eos and require_eos:
        return None
    if row:
        return None
    if not rows or any(len(candidate) != len(rows[0]) for candidate in rows):
        return None
    return rows


class ARCMetrics:
    """Token- and task-weighted metric accumulator for ARC completions."""

    def __init__(self) -> None:
        self.loss_sum = 0.0
        self.token_count = 0
        self.correct_tokens = 0
        self.completion_exact = 0
        self.teacher_forced_examples = 0
        self.generated_examples = 0
        self.valid_grids = 0
        self.generated_tasks: set[str] = set()
        self.generated_query_counts: dict[str, int] = defaultdict(int)
        self.expected_query_counts: dict[str, int] = {}
        self.grid_exact = 0
        self.shape_exact = 0
        self.cell_correct = 0
        self.cell_total = 0
        self.malformed = 0
        self.task_queries: dict[str, list[bool]] = defaultdict(list)

    def update_teacher_forced(self, logits: torch.Tensor, target_ids: torch.Tensor) -> None:
        """Accumulate loss and exact completion metrics from decoder-aligned targets."""
        if logits.shape[:2] != target_ids.shape:
            raise ValueError("logits and target_ids must have matching batch and decoder dimensions")
        valid = target_ids.ne(IGNORE_INDEX)
        if not valid.any():
            return
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logits in teacher-forced metrics")
        log_probs = F.log_softmax(logits, dim=-1)
        self.loss_sum += float(-log_probs[valid, target_ids[valid]].sum().detach().cpu())
        predictions = logits.argmax(dim=-1)
        self.correct_tokens += int(predictions[valid].eq(target_ids[valid]).sum().item())
        self.token_count += int(valid.sum().item())
        for row in range(target_ids.shape[0]):
            row_valid = valid[row]
            if row_valid.any():
                self.teacher_forced_examples += 1
                self.completion_exact += int(torch.equal(predictions[row, row_valid], target_ids[row, row_valid]))

    def update_generated(self, task_id: str, predicted_tokens: Iterable[int], target_grid: list[list[int]]) -> None:
        """Score one autoregressive completion against its query grid."""
        self.generated_examples += 1
        self.generated_tasks.add(task_id)
        self.generated_query_counts[task_id] += 1
        predicted_grid = decode_arc_grid(predicted_tokens)
        if predicted_grid is None:
            self.malformed += 1
            self.task_queries[task_id].append(False)
            return
        self.valid_grids += 1
        exact = predicted_grid == target_grid
        self.grid_exact += int(exact)
        self.task_queries[task_id].append(exact)
        if len(predicted_grid) == len(target_grid) and all(
            len(predicted_row) == len(target_row)
            for predicted_row, target_row in zip(predicted_grid, target_grid)
        ):
            self.shape_exact += 1
            self.cell_total += sum(len(row) for row in target_grid)
            self.cell_correct += sum(
                predicted_value == target_value
                for predicted_row, target_row in zip(predicted_grid, target_grid)
                for predicted_value, target_value in zip(predicted_row, target_row)
            )

    def as_dict(self) -> dict[str, float]:
        teacher_forced_loss = self.loss_sum / self.token_count if self.token_count else float("inf")
        generated_count = self.generated_examples
        metrics = {
            "loss": teacher_forced_loss,
            "perplexity": math.exp(min(20.0, teacher_forced_loss)) if self.token_count else float("inf"),
            "token_accuracy": self.correct_tokens / self.token_count if self.token_count else 0.0,
            "completion_exact": self.completion_exact / self.teacher_forced_examples if self.teacher_forced_examples else 0.0,
            "teacher_forced_examples": float(self.teacher_forced_examples),
            "generated_examples": float(generated_count),
            "examples": float(self.teacher_forced_examples),
            "supervised_tokens": float(self.token_count),
            "valid_grid_rate": self.valid_grids / generated_count if generated_count else 0.0,
            "grid_exact": self.grid_exact / generated_count if generated_count else 0.0,
            "shape_accuracy": self.shape_exact / generated_count if generated_count else 0.0,
            "cell_accuracy": self.cell_correct / self.cell_total if self.cell_total else 0.0,
            "malformed_outputs": float(self.malformed),
            "task_exact": (
                sum(all(query_results) for query_results in self.task_queries.values()) / len(self.task_queries)
                if self.task_queries else 0.0
            ),
        }
        return metrics

    def validate(self, *, require_teacher_forced: bool = True, require_generated: bool = True) -> None:
        if require_teacher_forced and (not self.token_count or not self.teacher_forced_examples):
            raise ValueError("evaluation produced no teacher-forced supervised examples")
        if require_generated and not self.generated_examples:
            raise ValueError("evaluation produced no generated examples")
        metrics = self.as_dict()
        invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
        if invalid:
            raise FloatingPointError(f"non-finite metrics: {', '.join(invalid)}")
        incomplete_tasks = [task_id for task_id, count in self.generated_query_counts.items()
                            if self.expected_query_counts.get(task_id, count) != count]
        if incomplete_tasks:
            raise ValueError(f"generated evaluation incomplete for tasks: {', '.join(incomplete_tasks[:5])}")

    def set_expected_query_counts(self, task_ids: Iterable[str]) -> None:
        for task_id in task_ids:
            self.expected_query_counts[task_id] = self.expected_query_counts.get(task_id, 0) + 1


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: batch[key].to(device)
        for key in ("input_ids", "attention_mask", "labels", "prefix_lengths", "target_ids")
    }


def _generation_inputs(tensors: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    """Build a right-padded, prompt-only batch and its shared decode cap.

    ``input_ids`` in an ARC training batch contains the target continuation, so
    it must never be forwarded directly to ``generate``.  Keeping the prompt
    batch shape separate also makes the returned-sequence slice unambiguous.
    """
    prefix_lengths = tensors["prefix_lengths"].to(dtype=torch.long)
    if prefix_lengths.ndim != 1 or prefix_lengths.numel() != tensors["input_ids"].shape[0]:
        raise ValueError("ARC generation requires one prefix length per batch row")
    if (prefix_lengths < 1).any():
        raise ValueError("ARC generation requires non-empty prefixes")
    prompt_width = int(prefix_lengths.max().item())
    if prompt_width > tensors["input_ids"].shape[1]:
        raise ValueError("ARC prefix length exceeds collated input width")
    target_lengths = tensors["target_ids"].ne(IGNORE_INDEX).sum(dim=-1)
    if not target_lengths.numel() or (target_lengths < 1).any():
        raise ValueError("ARC generation requires at least one target token per row")
    input_ids = tensors["input_ids"].new_full(
        (prefix_lengths.numel(), prompt_width), PAD_TOKEN
    )
    attention_mask = tensors["attention_mask"].new_zeros(input_ids.shape)
    for row, prefix_length in enumerate(prefix_lengths.tolist()):
        input_ids[row, :prefix_length] = tensors["input_ids"][row, :prefix_length]
        attention_mask[row, :prefix_length] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "prefix_lengths": prefix_lengths,
    }, int(target_lengths.max().item())


def _generated_completions(
    model: torch.nn.Module,
    generation_inputs: dict[str, torch.Tensor],
    max_new_tokens: int,
) -> list[list[int]]:
    generated = model.generate(
        **generation_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        eos_token_id=EOS_TOKEN,
        pad_token_id=PAD_TOKEN,
    )
    prompt_width = generation_inputs["input_ids"].shape[1]
    if (
        generated.ndim != 2
        or generated.shape[0] != generation_inputs["input_ids"].shape[0]
        or generated.shape[1] < prompt_width
    ):
        raise ValueError("generation returned an invalid sequence shape")
    return [row[prompt_width:].tolist() for row in generated]


def _assert_finite_outputs(
    outputs: Any,
    *,
    namespace: str,
    batch_index: int,
    task_ids: Iterable[str],
    supervised_tokens: int,
) -> None:
    context = (
        f"namespace={namespace}, batch={batch_index}, "
        f"task_ids={','.join(task_ids)}, supervised_tokens={supervised_tokens}"
    )
    if outputs.loss is None:
        raise ValueError(f"evaluation model output is missing loss at {context}")
    if not torch.isfinite(outputs.loss):
        raise FloatingPointError(f"non-finite evaluation loss at {context}")
    if not torch.isfinite(outputs.logits).all():
        maximum = outputs.logits.detach().abs().nan_to_num(posinf=float("inf")).max().item()
        raise FloatingPointError(
            f"non-finite evaluation logits at {context}, max_abs_logit={maximum}"
        )


def _write_metrics(writer: Any, namespace: str, metrics: dict[str, float], step: int) -> None:
    invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
    if invalid:
        raise FloatingPointError(f"non-finite {namespace} metrics: {', '.join(invalid)}")
    if writer is None:
        return
    for name, value in metrics.items():
        writer.add_scalar(f"{namespace}/{name}", value, step)


def _gradient_accumulation_size(dataloader: Iterable[dict[str, Any]], batch_index: int, steps: int) -> int:
    if not hasattr(dataloader, "__len__"):
        return steps
    total_batches = len(dataloader)
    group_start = batch_index - batch_index % steps
    return min(steps, total_batches - group_start)


def _diagnostics(writer: Any, outputs: Any, step: int) -> None:
    if writer is None:
        return
    if outputs.active_loops is not None:
        writer.add_scalar("diagnostics/active_loops", outputs.active_loops.float().mean().item(), step)
    if outputs.halting_probs is not None:
        writer.add_scalar("diagnostics/halting_probability", outputs.halting_probs.float().mean().item(), step)
    if outputs.loop_values is not None:
        writer.add_scalar("diagnostics/uncalibrated_value", outputs.loop_values.float().mean().item(), step)
    if outputs.loop_uncertainties is not None:
        writer.add_scalar("diagnostics/uncalibrated_uncertainty", outputs.loop_uncertainties.float().mean().item(), step)
    if outputs.router_counts:
        counts = sum(outputs.router_counts).float()
        total = counts.sum().clamp_min(1)
        for expert, fraction in enumerate(counts / total):
            writer.add_scalar(f"router/expert_{expert}_fraction", fraction.item(), step)


def _add_counts(accumulated: tuple[torch.Tensor, ...] | None, counts: tuple[torch.Tensor, ...] | None):
    if not counts:
        return accumulated
    detached = tuple(count.detach() for count in counts)
    if accumulated is None:
        return detached
    return tuple(left + right for left, right in zip(accumulated, detached))


def _distributed_router_counts(counts: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return counts
    reduced = tuple(item.clone() for item in counts)
    for item in reduced:
        torch.distributed.all_reduce(item, op=torch.distributed.ReduceOp.SUM)
    return reduced


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def synchronize_gradients(model: torch.nn.Module) -> None:
    """Synchronize graph-training gradients computed outside DDP.forward()."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    world_size = torch.distributed.get_world_size()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
        parameter.grad.div_(world_size)


def _hide_nonzero_rank_progress() -> bool:
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_rank() != 0
    )


def train_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    writer: Any = None,
    epoch: int = 0,
    global_step: int = 0,
    max_steps: int | None = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
    scheduler: Any = None,
) -> tuple[int, dict[str, float]]:
    """Train for one epoch, returning effective optimizer steps and aggregate metrics."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    model.train()
    metrics = ARCMetrics()
    router_counts = None
    optimizer.zero_grad(set_to_none=True)
    start = time.monotonic()
    microbatches = 0
    input_token_count = 0
    progress_total = len(dataloader) if hasattr(dataloader, "__len__") else None
    if max_steps is not None and progress_total is not None:
        remaining_steps = max(max_steps - global_step, 0)
        progress_total = min(progress_total, remaining_steps * gradient_accumulation_steps)
    progress = tqdm(
        dataloader,
        total=progress_total,
        desc=f"Train epoch {epoch + 1}",
        unit="batch",
        dynamic_ncols=True,
        disable=_hide_nonzero_rank_progress(),
        position=1,
        leave=False,
    )
    for batch_index, batch in enumerate(progress):
        tensors = _to_device(batch, device)
        input_token_count += int(tensors["attention_mask"].sum().item())
        outputs = model(**{key: tensors[key] for key in ("input_ids", "attention_mask", "labels", "prefix_lengths")}, use_cache=False)
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            raise FloatingPointError(f"non-finite training loss at epoch={epoch}, batch={batch_index}")
        accumulation_size = _gradient_accumulation_size(dataloader, batch_index, gradient_accumulation_steps)
        (outputs.loss / accumulation_size).backward()
        metrics.update_teacher_forced(outputs.logits.detach(), tensors["target_ids"])
        router_counts = _add_counts(router_counts, outputs.router_counts)
        microbatches += 1
        should_step = microbatches % gradient_accumulation_steps == 0
        is_last = hasattr(dataloader, "__len__") and batch_index + 1 == len(dataloader)
        if not should_step and not is_last:
            continue
        if max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at global_step={global_step}")
            if writer is not None:
                writer.add_scalar("train/gradient_norm", grad_norm.item(), global_step)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        update_router_biases(_unwrap_model(model), _distributed_router_counts(router_counts or ()))
        router_counts = None
        optimizer.zero_grad(set_to_none=True)
        if writer is not None:
            writer.add_scalar("train/loss_step", outputs.loss.item(), global_step)
            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        _diagnostics(writer, outputs, global_step)
        global_step += 1
        progress.set_postfix(
            loss=f"{outputs.loss.item():.4f}",
            step=global_step,
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )
        if max_steps is not None and global_step >= max_steps:
            break
    aggregate = metrics.as_dict()
    aggregate["input_tokens"] = float(input_token_count)
    aggregate["tokens_per_second"] = metrics.token_count / max(time.monotonic() - start, 1e-9)
    _write_metrics(writer, "train", aggregate, global_step)
    return global_step, aggregate


def train_graph_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    search_config: CordSearchConfig,
    replay_buffer: CordGraphReplayBuffer | None = None,
    replay_batch_size: int = 8,
    writer: Any = None,
    epoch: int = 0,
    global_step: int = 0,
    max_steps: int | None = None,
    scheduler: Any = None,
    max_grad_norm: float | None = None,
    lm_weight: float = 1.0,
    policy_weight: float = 1.0,
    local_bptt_weight: float = 0.1,
    graph_decoder_weight: float = 0.1,
    awr_temperature: float = 0.5,
    max_advantage_weight: float = 20.0,
    decode_max_new_tokens: int = 1024,
    ppo_clip: float | None = None,
    reference_model: torch.nn.Module | None = None,
    kl_weight: float = 0.0,
) -> tuple[int, dict[str, float], CordGraphReplayBuffer]:
    """Collect verified training rollouts and optimize SFT + AWR + local BPTT.

    This is intentionally a training-only API.  ARC targets serve as offline
    terminal verifiers after policy selection; validation/test evaluation uses
    :func:`compare_direct_and_graph`, which reports selected and oracle outcomes
    separately and performs no parameter update.
    """

    from .arc_verifier import ARCDecodedVerifier

    if replay_batch_size < 1:
        raise ValueError("replay_batch_size must be positive")
    if decode_max_new_tokens < 1:
        raise ValueError("decode_max_new_tokens must be positive")
    if min(lm_weight, policy_weight, local_bptt_weight, graph_decoder_weight, kl_weight) < 0.0:
        raise ValueError("graph training loss weights must be non-negative")
    if ppo_clip is not None and ppo_clip <= 0.0:
        raise ValueError("ppo_clip must be positive")
    if kl_weight and reference_model is None:
        raise ValueError("KL regularization requires a frozen reference_model")
    replay_buffer = CordGraphReplayBuffer() if replay_buffer is None else replay_buffer
    metrics = ARCMetrics()
    graph_loss_sum = 0.0
    policy_loss_sum = 0.0
    local_loss_sum = 0.0
    decoder_loss_sum = 0.0
    kl_loss_sum = 0.0
    selected_exact = 0
    oracle_exact = 0
    rollout_count = 0
    graph_steps = 0
    input_token_count = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        dataloader, desc=f"Graph train epoch {epoch + 1}", unit="batch",
        dynamic_ncols=True, disable=_hide_nonzero_rank_progress(), position=1, leave=False,
    )
    for batch_index, batch in enumerate(progress):
        tensors = _to_device(batch, device)
        input_token_count += int(tensors["attention_mask"].sum().item())
        rollout_transitions = []
        graph_model = _unwrap_model(model)
        graph_model.eval()
        for row, task_id in enumerate(batch["task_ids"]):
            prefix_length = int(tensors["prefix_lengths"][row].item())
            target_length = int(tensors["target_ids"][row].ne(IGNORE_INDEX).sum().item())
            if prefix_length < 1 or target_length < 1:
                raise ValueError(f"invalid graph-training sample {task_id}")
            prompt = tensors["input_ids"][row : row + 1, :prefix_length]
            search = graph_model.search(
                prompt,
                prefix_lengths=prompt.new_tensor([prefix_length]),
                search_config=search_config,
                verifier=ARCDecodedVerifier(batch["target_grids"][row], shaping=True),
                max_new_tokens=decode_max_new_tokens,
                task_id=task_id,
            )
            if search.replay_transitions:
                decoder_targets = tensors["target_ids"][row].detach()
                for transition in search.replay_transitions:
                    if transition.terminal:
                        transition.decoder_target_ids = decoder_targets
                rollout_transitions.extend(search.replay_transitions)
            if search.selected_index is None or search.verified is None:
                raise RuntimeError("training rollout is missing policy-selection diagnostics")
            selected_exact += int(bool(search.verified[int(search.selected_index)]))
            oracle_exact += int(any(search.verified))
            rollout_count += 1
        replay_buffer.extend(rollout_transitions)
        replay_sample = (
            list(rollout_transitions)
            if ppo_clip is not None
            else replay_buffer.sample(replay_batch_size)
        )
        if not replay_sample:
            raise RuntimeError("graph rollout produced no replayable transitions")

        graph_model.train()
        outputs = graph_model(
            **{key: tensors[key] for key in ("input_ids", "attention_mask", "labels", "prefix_lengths")},
            use_cache=False,
        )
        if outputs.loss is None or not torch.isfinite(outputs.loss):
            raise FloatingPointError(f"non-finite SFT anchor loss at graph epoch={epoch}, batch={batch_index}")
        controller = getattr(getattr(graph_model, "model", graph_model), "state_controller")
        policy_loss, policy_breakdown = compute_graph_policy_loss(
            controller,
            replay_sample,
            device=device,
            awr_temperature=awr_temperature,
            max_advantage_weight=max_advantage_weight,
            ppo_clip=ppo_clip,
        )
        local_loss = compute_local_transition_loss(
            graph_model,
            replay_sample,
            awr_temperature=awr_temperature,
            max_advantage_weight=max_advantage_weight,
        )
        decoder_loss = compute_graph_decoder_loss(
            graph_model,
            replay_sample,
            awr_temperature=awr_temperature,
            max_advantage_weight=max_advantage_weight,
        )
        kl_loss = outputs.loss.new_zeros(())
        if reference_model is not None and kl_weight:
            reference_model.eval()
            with torch.no_grad():
                reference_outputs = reference_model(
                    **{key: tensors[key] for key in ("input_ids", "attention_mask", "labels", "prefix_lengths")},
                    use_cache=False,
                )
            kl_mask = tensors["target_ids"].ne(IGNORE_INDEX)
            kl_loss = F.kl_div(
                outputs.logits[kl_mask].log_softmax(dim=-1),
                reference_outputs.logits[kl_mask].softmax(dim=-1),
                reduction="batchmean",
            ).clamp_min(0.0)
        total_loss = (
            lm_weight * outputs.loss
            + policy_weight * policy_loss
            + local_bptt_weight * local_loss
            + graph_decoder_weight * decoder_loss
            + kl_weight * kl_loss
        )
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"non-finite graph training loss at epoch={epoch}, batch={batch_index}")
        total_loss.backward()
        synchronize_gradients(graph_model)
        if max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(graph_model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite graph gradient norm at global_step={global_step}")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        update_router_biases(graph_model, _distributed_router_counts(outputs.router_counts or ()))
        optimizer.zero_grad(set_to_none=True)
        metrics.update_teacher_forced(outputs.logits.detach(), tensors["target_ids"])
        graph_loss_sum += float(total_loss.detach().item())
        policy_loss_sum += float(policy_loss.detach().item())
        local_loss_sum += float(local_loss.detach().item())
        decoder_loss_sum += float(decoder_loss.detach().item())
        kl_loss_sum += float(kl_loss.detach().item())
        if writer is not None:
            writer.add_scalar("graph/total_loss", total_loss.item(), global_step)
            writer.add_scalar("graph/policy_loss", policy_loss.item(), global_step)
            writer.add_scalar("graph/local_bptt_loss", local_loss.item(), global_step)
            writer.add_scalar("graph/decoder_grounding_loss", decoder_loss.item(), global_step)
            writer.add_scalar("graph/reference_kl", kl_loss.item(), global_step)
            writer.add_scalar("graph/replay_size", len(replay_buffer), global_step)
            for name, value in policy_breakdown.items():
                writer.add_scalar(f"graph/{name}", value.item(), global_step)
        global_step += 1
        graph_steps += 1
        progress.set_postfix(
            loss=f"{total_loss.item():.4f}",
            selected=f"{selected_exact / rollout_count:.3f}",
            oracle=f"{oracle_exact / rollout_count:.3f}",
        )
        if max_steps is not None and global_step >= max_steps:
            break
    if rollout_count == 0:
        raise ValueError("graph training produced no rollouts")
    steps = max(graph_steps, 1)
    aggregate = metrics.as_dict()
    aggregate.update({
        "graph_total_loss": graph_loss_sum / steps,
        "graph_policy_loss": policy_loss_sum / steps,
        "graph_local_bptt_loss": local_loss_sum / steps,
        "graph_decoder_grounding_loss": decoder_loss_sum / steps,
        "graph_reference_kl": kl_loss_sum / steps,
        "graph_policy_algorithm": 1.0 if ppo_clip is not None else 0.0,
        "selected_solution_accuracy": selected_exact / rollout_count,
        "oracle_success_at_n": oracle_exact / rollout_count,
        "graph_rollouts": float(rollout_count),
        "replay_size": float(len(replay_buffer)),
        "input_tokens": float(input_token_count),
    })
    _write_metrics(writer, "graph_train", aggregate, global_step)
    return global_step, aggregate, replay_buffer


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, Any]],
    device: torch.device,
    writer: Any = None,
    global_step: int = 0,
    namespace: str = "val",
    max_steps: int | None = None,
    generate: bool = True,
) -> dict[str, float]:
    """Run teacher-forced and, by default, autoregressive ARC evaluation."""
    model.eval()
    metrics = ARCMetrics()
    progress_total = len(dataloader) if hasattr(dataloader, "__len__") else None
    if max_steps is not None and progress_total is not None:
        progress_total = min(progress_total, max_steps)
    progress = tqdm(
        dataloader,
        total=progress_total,
        desc=f"Evaluate {namespace}",
        unit="batch",
        dynamic_ncols=True,
        disable=_hide_nonzero_rank_progress(),
        position=1,
        leave=False,
    )
    for batch_index, batch in enumerate(progress):
        tensors = _to_device(batch, device)
        outputs = model(**{key: tensors[key] for key in ("input_ids", "attention_mask", "labels", "prefix_lengths")}, use_cache=False)
        _assert_finite_outputs(
            outputs,
            namespace=namespace,
            batch_index=batch_index,
            task_ids=batch["task_ids"],
            supervised_tokens=int(tensors["target_ids"].ne(IGNORE_INDEX).sum().item()),
        )
        metrics.update_teacher_forced(outputs.logits, tensors["target_ids"])
        if generate:
            metrics.set_expected_query_counts(batch["task_ids"])
            generation_inputs, max_new_tokens = _generation_inputs(tensors)
            completions = _generated_completions(model, generation_inputs, max_new_tokens)
            for task_id, completion, target_grid in zip(
                batch["task_ids"], completions, batch["target_grids"], strict=True
            ):
                metrics.update_generated(task_id, completion, target_grid)
        _diagnostics(writer, outputs, global_step)
        running_loss = metrics.loss_sum / metrics.token_count if metrics.token_count else float("inf")
        progress.set_postfix(
            loss=f"{running_loss:.4f}",
            generated_exact=f"{metrics.grid_exact / metrics.generated_examples:.3f}" if metrics.generated_examples else "n/a",
        )
        if max_steps is not None and batch_index + 1 >= max_steps:
            break
    metrics.validate(require_teacher_forced=True, require_generated=generate)
    aggregate = metrics.as_dict()
    _write_metrics(writer, namespace, aggregate, global_step)
    return aggregate


def is_better_validation(current: dict[str, float], best: dict[str, float] | None) -> bool:
    """Compare validation metrics by ARC task quality, then teacher-forced loss."""
    if best is None:
        return True
    return (current["task_exact"], current["grid_exact"], -current["loss"]) > (
        best["task_exact"], best["grid_exact"], -best["loss"]
    )


def validate_metrics(metrics: dict[str, float]) -> None:
    required = {
        "loss", "perplexity", "token_accuracy", "completion_exact",
        "teacher_forced_examples", "generated_examples", "supervised_tokens",
        "valid_grid_rate", "grid_exact", "shape_accuracy", "cell_accuracy",
        "malformed_outputs", "task_exact",
    }
    missing = sorted(required - metrics.keys())
    if missing:
        raise ValueError(f"incomplete evaluation metrics: {', '.join(missing)}")
    invalid = [name for name, value in metrics.items() if not math.isfinite(value)]
    if invalid:
        raise FloatingPointError(f"non-finite metrics: {', '.join(invalid)}")
    for name in ("teacher_forced_examples", "generated_examples", "supervised_tokens"):
        if metrics[name] <= 0:
            raise ValueError(f"evaluation metric {name} must be positive")


# Compatibility alias for older scripts.
def validate(*args: Any, **kwargs: Any) -> float:
    metrics = evaluate(*args, namespace="val", **kwargs)
    return metrics["loss"]


__all__ = [
    "ARCMetrics", "decode_arc_grid", "evaluate", "is_better_validation", "train_epoch",
    "train_graph_epoch", "validate", "validate_metrics",
]
