"""Train the registered sub-50M CoRD prototype on public ARC-AGI-2."""

from __future__ import annotations

import argparse
import json
import random
import os
import subprocess
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cord import (
    CordConfig,
    CordForCausalLM,
    CordGraphReplayBuffer,
    CordSearchConfig,
    build_cord_optimizer_param_groups,
    build_graph_optimizer_param_groups,
)
from dataset.data import ARC_VOCAB_SIZE, ARCDataset, collate_fn, discover_arc_tasks, split_arc_training_files
from trainer.arc_search_evaluation import compare_direct_and_graph
from trainer.trainer import evaluate, is_better_validation, train_epoch, train_graph_epoch, validate_metrics


def unique_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in {id(item): item for item in model.parameters()}.values())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog="Monitor locally: tensorboard --logdir runs --host 127.0.0.1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "ARC-AGI-2")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cord-50m.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "arc_agi_2")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "runs" / "arc_agi_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--strategy",
        choices=("single", "ddp", "fsdp", "deepspeed_stage_3"),
        default="single",
        help="Parallelism strategy. DDP is implemented natively; FSDP/DeepSpeed fail explicitly until supported.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=None,
        metavar="GPU",
        help="Physical CUDA device indices, for example: --devices 0 1.",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=1,
        help="Additional task-consistent augmented copies per train query per epoch (plus canonical copy).",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=None, help="Refuse samples longer than this; never truncates.")
    parser.add_argument("--smoke-optimizer-steps", type=int, default=None, help="Explicitly cap optimizer steps for smoke runs.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Load a saved model checkpoint before training or --eval-only.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--graph-training-epochs", type=int, default=0, help="AWR/search-distillation epochs after SFT.")
    parser.add_argument("--ppo-training-epochs", type=int, default=0, help="On-policy PPO epochs after AWR calibration.")
    parser.add_argument("--ppo-clip", type=float, default=0.2)
    parser.add_argument("--ppo-kl-weight", type=float, default=0.01)
    parser.add_argument("--graph-max-expansions", type=int, default=4)
    parser.add_argument("--graph-beam-size", type=int, default=4)
    parser.add_argument("--graph-verified-leaves", type=int, default=4)
    parser.add_argument("--graph-decode-max-new-tokens", type=int, default=1024)
    parser.add_argument("--graph-replay-capacity", type=int, default=4096)
    parser.add_argument("--graph-replay-batch-size", type=int, default=8)
    parser.add_argument("--controller-learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--awr-temperature", type=float, default=0.5)
    parser.add_argument("--graph-policy-weight", type=float, default=1.0)
    parser.add_argument("--local-bptt-weight", type=float, default=0.1)
    parser.add_argument("--graph-decoder-weight", type=float, default=0.1)
    return parser.parse_args(argv)


def validate_parallel_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.devices is not None and (any(index < 0 for index in args.devices) or len(set(args.devices)) != len(args.devices)):
        raise ValueError("--devices must contain unique non-negative GPU indices")
    if args.devices is not None and "LOCAL_RANK" not in os.environ and torch.cuda.is_available():
        if max(args.devices) >= torch.cuda.device_count():
            raise ValueError(f"--devices references an unavailable GPU; found {torch.cuda.device_count()} CUDA devices")
    if args.strategy in {"fsdp", "deepspeed_stage_3"}:
        raise NotImplementedError(
            f"strategy {args.strategy!r} is not implemented for CoRD graph replay; use --strategy ddp"
        )
    if args.strategy == "ddp" and (args.devices is None or len(args.devices) < 2):
        raise ValueError("--strategy ddp requires at least two entries in --devices")
    if args.strategy == "single" and args.devices is not None and len(args.devices) != 1:
        raise ValueError("--strategy single accepts exactly one device")


def launch_distributed_if_needed(args: argparse.Namespace) -> bool:
    """Relaunch this entrypoint under torchrun from a plain Python invocation."""

    if args.strategy != "ddp" or "LOCAL_RANK" in os.environ:
        return False
    visible = ",".join(str(index) for index in args.devices)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = visible
    environment["CORD_RUN_ID"] = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-seed{args.seed}"
    environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(len(args.devices)),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    subprocess.run(command, check=True, env=environment)
    return True


def initialize_parallel(args: argparse.Namespace) -> tuple[torch.device, int, int]:
    if args.strategy == "ddp":
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA for the configured NCCL backend")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return torch.device("cuda", local_rank), torch.distributed.get_rank(), torch.distributed.get_world_size()
    if args.devices is not None:
        torch.cuda.set_device(args.devices[0])
        return torch.device("cuda", args.devices[0]), 0, 1
    return torch.device(args.device), 0, 1


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "module", model)


def reduce_training_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    """Aggregate rank-local training diagnostics into one global report."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return metrics
    summed_metrics = {
        "teacher_forced_examples", "generated_examples", "supervised_tokens",
        "malformed_outputs", "graph_rollouts", "tokens_per_second", "input_tokens",
    }
    reduced = {}
    world_size = torch.distributed.get_world_size()
    for name, value in metrics.items():
        tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        if name not in summed_metrics:
            tensor.div_(world_size)
        reduced[name] = float(tensor.item())
    return reduced


def build_training_plan(
    train_lengths: dict[str, int],
    *,
    sft_epochs: int,
    awr_epochs: int,
    ppo_epochs: int,
    steps_per_sft_epoch: int,
    steps_per_graph_epoch: int,
    verified_leaves: int,
    decode_max_new_tokens: int,
) -> dict[str, object]:
    graph_epochs = awr_epochs + ppo_epochs
    total_epochs = sft_epochs + graph_epochs
    return {
        "segments": [
            {"name": "SFT", "epochs": sft_epochs, "optimizer_steps": sft_epochs * steps_per_sft_epoch},
            {"name": "AWR", "epochs": awr_epochs, "optimizer_steps": awr_epochs * steps_per_graph_epoch},
            {"name": "PPO", "epochs": ppo_epochs, "optimizer_steps": ppo_epochs * steps_per_graph_epoch},
        ],
        "tokens_per_epoch": {
            "input": train_lengths["total_tokens"],
            "prefix": train_lengths["prefix_tokens"],
            "supervised": train_lengths["supervised_tokens"],
        },
        "scheduled_tokens": {
            "input": train_lengths["total_tokens"] * total_epochs,
            "supervised": train_lengths["supervised_tokens"] * total_epochs,
            "graph_decode_upper_bound": (
                train_lengths["samples"] * graph_epochs * verified_leaves * decode_max_new_tokens
            ),
        },
        "total_optimizer_steps": sft_epochs * steps_per_sft_epoch + graph_epochs * steps_per_graph_epoch,
    }


def print_training_plan(plan: dict[str, object], *, world_size: int, batch_size: int, accumulation: int) -> None:
    segment_text = " | ".join(
        f"{segment['name']} {segment['epochs']} ep / {segment['optimizer_steps']} steps"
        for segment in plan["segments"]
    )
    tokens = plan["tokens_per_epoch"]
    scheduled = plan["scheduled_tokens"]
    print("\n=== CoRD training manifest ===")
    print(f"Timeline: [{segment_text}]")
    print(f"Batch: {batch_size}/GPU x {world_size} GPU x {accumulation} accumulation = {batch_size * world_size * accumulation}")
    print(
        f"Tokens/epoch: {tokens['input']:,} input "
        f"({tokens['prefix']:,} prefix + {tokens['supervised']:,} supervised)"
    )
    print(f"Scheduled training tokens: {scheduled['input']:,} input; {scheduled['supervised']:,} supervised")
    print(f"Graph decode token upper bound: {scheduled['graph_decode_upper_bound']:,}")
    print(f"Optimizer steps: {plan['total_optimizer_steps']:,}\n")


class TrainingTimeline:
    """One rank-zero progress bar spanning the SFT, AWR, and PPO segments."""

    def __init__(self, plan: dict[str, object], enabled: bool):
        self.enabled = enabled
        self.completed_steps = 0
        self.processed_tokens = 0
        self.progress = tqdm(
            total=int(plan["total_optimizer_steps"]),
            desc="Overall SFT>AWR>PPO",
            unit="step",
            dynamic_ncols=True,
            position=0,
        ) if enabled else None

    def advance(self, global_step: int, phase: str, epoch: int, epochs: int, input_tokens: float) -> None:
        if not self.enabled:
            return
        delta = max(global_step - self.completed_steps, 0)
        self.completed_steps = global_step
        self.processed_tokens += int(input_tokens)
        self.progress.update(delta)
        self.progress.set_description(f"Overall [{phase} {epoch}/{epochs}]")
        self.progress.set_postfix(tokens=f"{self.processed_tokens:,}", refresh=True)

    def close(self) -> None:
        if self.progress is not None:
            self.progress.close()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float) -> LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        remaining = max(total_steps - warmup_steps, 1)
        return max(0.0, (total_steps - step) / remaining)

    return LambdaLR(optimizer, scale)


def main() -> None:
    args = parse_args()
    validate_parallel_args(args)
    if launch_distributed_if_needed(args):
        return
    device, rank, world_size = initialize_parallel(args)
    is_main_process = rank == 0
    if args.train_fraction != 0.8:
        raise ValueError("ARC audit contract requires --train-fraction 0.8")
    if args.graph_training_epochs < 0:
        raise ValueError("--graph-training-epochs cannot be negative")
    if args.ppo_training_epochs < 0:
        raise ValueError("--ppo-training-epochs cannot be negative")
    if args.ppo_training_epochs and not args.graph_training_epochs:
        raise ValueError("PPO requires at least one AWR graph-training epoch for calibration")
    if args.graph_training_epochs and not args.eval_only and args.epochs < 1 and args.resume_from is None:
        raise ValueError("graph training requires an SFT phase or --resume-from")
    seed_everything(args.seed + rank)
    training_files = discover_arc_tasks(args.data_dir / "training")
    test_files = discover_arc_tasks(args.data_dir / "evaluation")
    split = split_arc_training_files(training_files, seed=args.seed, train_fraction=args.train_fraction)
    if len(set(split.train_files) & set(split.validation_files)):
        raise RuntimeError("training and validation tasks overlap")
    run_id = os.environ.get("CORD_RUN_ID", f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-seed{args.seed}")
    output_dir = args.output_dir / run_id
    log_dir = args.log_dir / run_id
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=False)
    barrier()
    writer = SummaryWriter(log_dir=str(log_dir)) if is_main_process else None
    try:
        train_dataset = ARCDataset(
            split.train_files,
            max_length=args.max_length,
            augment=True,
            num_aug=args.num_aug,
            augmentation_seed=args.seed,
            split_name="train",
        )
        validation_dataset = ARCDataset(split.validation_files, max_length=args.max_length, augment=False, augmentation_seed=args.seed, split_name="validation")
        test_dataset = ARCDataset(test_files, max_length=args.max_length, augment=False, augmentation_seed=args.seed, split_name="test")
        loader = partial(collate_fn, max_length=args.max_length)
        generator = torch.Generator().manual_seed(args.seed)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        ) if world_size > 1 else None
        pin_memory = device.type == "cuda"
        persistent_workers = args.num_workers > 0
        train_loader = DataLoader(
            train_dataset,
            args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            generator=generator,
            num_workers=args.num_workers,
            collate_fn=loader,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        validation_loader = DataLoader(
            validation_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=loader,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=loader,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        train_lengths = train_dataset.lengths()
        validation_lengths = validation_dataset.lengths()
        test_lengths = test_dataset.lengths()
        config = CordConfig.from_json_file(args.config)
        if config.vocab_size < ARC_VOCAB_SIZE:
            raise ValueError(f"model vocabulary ({config.vocab_size}) cannot represent ARC tokens ({ARC_VOCAB_SIZE})")
        model = CordForCausalLM.from_pretrained(args.resume_from) if args.resume_from else CordForCausalLM(config)
        parameter_count = unique_parameter_count(model)
        if parameter_count >= 50_000_000:
            raise RuntimeError(f"CoRD model has {parameter_count:,} unique parameters; ARC audit requires <50,000,000")
        model.to(device)
        if args.strategy == "ddp":
            model = DistributedDataParallel(
                model,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=True,
            )
        raw_groups = build_cord_optimizer_param_groups(model, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW([
            {"params": group["params"], "weight_decay": group["weight_decay"]} for group in raw_groups
        ], lr=args.learning_rate)
        estimated_steps = max(1, (len(train_loader) * args.epochs + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
        scheduler = make_scheduler(optimizer, estimated_steps, args.warmup_ratio)
        steps_per_sft_epoch = max(
            1, (len(train_loader) + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
        )
        training_plan = build_training_plan(
            train_lengths,
            sft_epochs=args.epochs,
            awr_epochs=args.graph_training_epochs,
            ppo_epochs=args.ppo_training_epochs,
            steps_per_sft_epoch=steps_per_sft_epoch,
            steps_per_graph_epoch=len(train_loader),
            verified_leaves=args.graph_verified_leaves,
            decode_max_new_tokens=args.graph_decode_max_new_tokens,
        )
        graph_search_config = CordSearchConfig(
            max_expansions=args.graph_max_expansions,
            beam_size=args.graph_beam_size,
            max_verified_leaves=args.graph_verified_leaves,
            deterministic=True,
            seed=args.seed,
        )
        metadata = {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "parameter_count": parameter_count,
            "parallelism": {
                "strategy": args.strategy,
                "devices": args.devices,
                "world_size": world_size,
                "batch_size_per_device": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.batch_size * world_size * args.gradient_accumulation_steps,
            },
            "split_manifest_digest": split.manifest_digest,
            "train_tasks": len(split.train_files), "validation_tasks": len(split.validation_files), "test_tasks": len(test_files),
            "num_aug": args.num_aug,
            "train_lengths": train_lengths, "validation_lengths": validation_lengths, "test_lengths": test_lengths,
            "training_plan": training_plan,
            "evaluation_protocol": {
                "teacher_forced": True,
                "autoregressive_generation": True,
                "generation": {
                    "decode": "greedy",
                    "do_sample": False,
                    "num_beams": 1,
                    "use_cache": True,
                    "eos_token_id": 2,
                    "pad_token_id": 0,
                    "max_new_tokens": "largest non-padding target completion in each batch",
                },
                "checkpoint_selection": ["task_exact", "grid_exact", "-loss"],
                "state_graph": {
                    "training_algorithm": "AWR + Monte Carlo value + local BPTT",
                    "target_verifier_timing": "post-selection only",
                    "reported_metrics": ["selected_solution_accuracy", "oracle_success_at_n"],
                    "search_config": graph_search_config.__dict__,
                },
            },
        }
        if is_main_process:
            (output_dir / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
            writer.add_text("run/config", json.dumps(metadata, indent=2), 0)
            writer.add_scalar("system/parameter_count", parameter_count, 0)
            print(json.dumps(metadata, indent=2))
            print_training_plan(
                training_plan,
                world_size=world_size,
                batch_size=args.batch_size,
                accumulation=args.gradient_accumulation_steps,
            )
        timeline = TrainingTimeline(training_plan, enabled=is_main_process and not args.eval_only)
        global_step = 0
        best_metrics = None
        best_dir = output_dir / "best"
        if not args.eval_only:
            for epoch in range(args.epochs):
                train_dataset.set_epoch(epoch)
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                global_step, train_metrics = train_epoch(
                    model, train_loader, optimizer, device, writer, epoch, global_step,
                    max_steps=args.smoke_optimizer_steps, gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_grad_norm=args.max_grad_norm, scheduler=scheduler,
                )
                if is_main_process:
                    validation_metrics = evaluate(
                        unwrap_model(model), validation_loader, device, writer, global_step, namespace="val"
                    )
                    print(f"epoch={epoch} train={train_metrics} val={validation_metrics}")
                    validate_metrics(validation_metrics)
                    if is_better_validation(validation_metrics, best_metrics):
                        best_metrics = validation_metrics
                        unwrap_model(model).save_pretrained(best_dir)
                        torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "global_step": global_step, "validation_metrics": validation_metrics}, best_dir / "trainer_state.pt")
                        (output_dir / "best_validation_metrics.json").write_text(
                            json.dumps(validation_metrics, indent=2, allow_nan=False), encoding="utf-8"
                        )
                barrier()
                if args.smoke_optimizer_steps is not None and global_step >= args.smoke_optimizer_steps:
                    break
            if args.graph_training_epochs or args.ppo_training_epochs:
                if not best_dir.exists():
                    raise FileNotFoundError("graph phase requires a finite SFT checkpoint")
                sft_reference_dir = best_dir
                model = CordForCausalLM.from_pretrained(sft_reference_dir).to(device)
                graph_groups = build_graph_optimizer_param_groups(
                    model,
                    args.controller_learning_rate,
                    backbone_lr_scale=args.backbone_lr_scale,
                    weight_decay=args.weight_decay,
                )
                train_metrics = reduce_training_metrics(train_metrics, device)
                timeline.advance(global_step, "SFT", epoch + 1, args.epochs, train_metrics["input_tokens"])
                graph_optimizer = torch.optim.AdamW(graph_groups)
                graph_estimated_steps = max(
                    1, len(train_loader) * (args.graph_training_epochs + args.ppo_training_epochs)
                )
                graph_scheduler = make_scheduler(graph_optimizer, graph_estimated_steps, args.warmup_ratio)
                replay_buffer = CordGraphReplayBuffer(args.graph_replay_capacity)
                best_graph_metrics = None
                best_graph_dir = output_dir / "best_graph"
                graph_step_cap = (
                    global_step + args.smoke_optimizer_steps
                    if args.smoke_optimizer_steps is not None else None
                )
                reference_model = None
                if args.ppo_training_epochs:
                    reference_model = CordForCausalLM.from_pretrained(sft_reference_dir).to(device).eval()
                    for parameter in reference_model.parameters():
                        parameter.requires_grad_(False)
                phase_specs = [
                    ("awr", args.graph_training_epochs, None, None, 0.0),
                    ("ppo", args.ppo_training_epochs, args.ppo_clip, reference_model, args.ppo_kl_weight),
                ]
                graph_epoch_index = 0
                stop_graph = False
                for phase_name, phase_epochs, ppo_clip, phase_reference, kl_weight in phase_specs:
                    for phase_epoch in range(phase_epochs):
                        train_dataset.set_epoch(args.epochs + graph_epoch_index)
                        if train_sampler is not None:
                            train_sampler.set_epoch(args.epochs + graph_epoch_index)
                        global_step, graph_train_metrics, replay_buffer = train_graph_epoch(
                            model,
                            train_loader,
                            graph_optimizer,
                            device,
                            search_config=graph_search_config,
                            replay_buffer=replay_buffer,
                            replay_batch_size=args.graph_replay_batch_size,
                            writer=writer,
                            epoch=graph_epoch_index,
                            global_step=global_step,
                            max_steps=graph_step_cap,
                            scheduler=graph_scheduler,
                            max_grad_norm=args.max_grad_norm,
                            policy_weight=args.graph_policy_weight,
                            local_bptt_weight=args.local_bptt_weight,
                            graph_decoder_weight=args.graph_decoder_weight,
                            awr_temperature=args.awr_temperature,
                            decode_max_new_tokens=args.graph_decode_max_new_tokens,
                            ppo_clip=ppo_clip,
                            reference_model=phase_reference,
                            kl_weight=kl_weight,
                        )
                        graph_train_metrics = reduce_training_metrics(graph_train_metrics, device)
                        timeline.advance(
                            global_step, phase_name.upper(), phase_epoch + 1, phase_epochs,
                            graph_train_metrics["input_tokens"],
                        )
                        if is_main_process:
                            validation_metrics = evaluate(
                                model, validation_loader, device, writer, global_step,
                                namespace=f"{phase_name}_val",
                            )
                            graph_validation = compare_direct_and_graph(
                                model, validation_loader, device, search_config=graph_search_config,
                                decode_max_new_tokens=args.graph_decode_max_new_tokens,
                            )
                            candidate = {**validation_metrics, **graph_validation, "phase": phase_name}
                            print(f"phase={phase_name} epoch={phase_epoch} train={graph_train_metrics} val={candidate}")
                            graph_score = (
                                candidate["selected_solution_accuracy"], candidate["oracle_success_at_n"],
                                candidate["task_exact"], -candidate["loss"],
                            )
                            best_graph_score = None if best_graph_metrics is None else (
                                best_graph_metrics["selected_solution_accuracy"],
                                best_graph_metrics["oracle_success_at_n"], best_graph_metrics["task_exact"],
                                -best_graph_metrics["loss"],
                            )
                            if best_graph_score is None or graph_score > best_graph_score:
                                best_graph_metrics = candidate
                                model.save_pretrained(best_graph_dir)
                                torch.save(
                                    {"optimizer": graph_optimizer.state_dict(), "scheduler": graph_scheduler.state_dict(),
                                     "phase": phase_name, "epoch": phase_epoch, "global_step": global_step,
                                     "validation_metrics": candidate},
                                    best_graph_dir / "trainer_state.pt",
                                )
                                torch.save(replay_buffer.state_dict(), best_graph_dir / "graph_replay.pt")
                                (output_dir / "best_graph_validation_metrics.json").write_text(
                                    json.dumps(candidate, indent=2, allow_nan=False), encoding="utf-8"
                                )
                        barrier()
                        graph_epoch_index += 1
                        if graph_step_cap is not None and global_step >= graph_step_cap:
                            stop_graph = True
                            break
                    if stop_graph:
                        break
                if best_graph_dir.exists():
                    best_dir = best_graph_dir
        barrier()
        if is_main_process:
            checkpoint_dir = args.resume_from if args.eval_only and args.resume_from else best_dir
            if checkpoint_dir is None or not checkpoint_dir.exists():
                raise FileNotFoundError("no finite validation checkpoint exists; supply --resume-from for --eval-only")
            model = CordForCausalLM.from_pretrained(checkpoint_dir).to(device)
            test_metrics = evaluate(model, test_loader, device, writer, global_step, namespace="test")
            validate_metrics(test_metrics)
            print(f"final_test={test_metrics}")
            graph_test_metrics = None
            if args.graph_training_epochs or args.ppo_training_epochs:
                graph_test_metrics = compare_direct_and_graph(
                    model, test_loader, device, search_config=graph_search_config,
                    decode_max_new_tokens=args.graph_decode_max_new_tokens,
                )
                print(f"final_graph_test={graph_test_metrics}")
            final_artifacts = {
                "checkpoint": str(checkpoint_dir),
                "selection": metadata["evaluation_protocol"]["checkpoint_selection"],
                "teacher_forced_and_generated": test_metrics,
                "state_graph": graph_test_metrics,
            }
            (output_dir / "final_test_metrics.json").write_text(json.dumps(final_artifacts, indent=2, allow_nan=False), encoding="utf-8")
            (output_dir / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
            if best_metrics is not None:
                validate_metrics(best_metrics)
                metadata["best_validation_metrics"] = best_metrics
                (output_dir / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
            writer.flush()
        timeline.close()
    finally:
        if "timeline" in locals():
            timeline.close()
        if writer is not None:
            writer.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
