"""Train the registered sub-50M CoRD prototype on public ARC-AGI-2."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cord import CordConfig, CordForCausalLM, build_cord_optimizer_param_groups
from dataset.data import ARC_VOCAB_SIZE, ARCDataset, collate_fn, discover_arc_tasks, split_arc_training_files
from trainer.trainer import evaluate, train_epoch


def unique_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in {id(item): item for item in model.parameters()}.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, epilog="Monitor locally: tensorboard --logdir runs --host 127.0.0.1")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "ARC-AGI-2")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "cord-50m.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "arc_agi_2")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "runs" / "arc_agi_2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
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
    return parser.parse_args()


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
    if args.train_fraction != 0.8:
        raise ValueError("ARC audit contract requires --train-fraction 0.8")
    seed_everything(args.seed)
    training_files = discover_arc_tasks(args.data_dir / "training")
    test_files = discover_arc_tasks(args.data_dir / "evaluation")
    split = split_arc_training_files(training_files, seed=args.seed, train_fraction=args.train_fraction)
    if len(set(split.train_files) & set(split.validation_files)):
        raise RuntimeError("training and validation tasks overlap")
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-seed{args.seed}"
    output_dir = args.output_dir / run_id
    log_dir = args.log_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(log_dir=str(log_dir))
    try:
        train_dataset = ARCDataset(split.train_files, max_length=args.max_length, augment=True, augmentation_seed=args.seed, split_name="train")
        validation_dataset = ARCDataset(split.validation_files, max_length=args.max_length, augment=True, augmentation_seed=args.seed, split_name="validation")
        test_dataset = ARCDataset(test_files, max_length=args.max_length, augment=False, augmentation_seed=args.seed, split_name="test")
        loader = partial(collate_fn, max_length=args.max_length)
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, generator=generator, num_workers=args.num_workers, collate_fn=loader)
        validation_loader = DataLoader(validation_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=loader)
        test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=loader)
        config = CordConfig.from_json_file(args.config)
        if config.vocab_size < ARC_VOCAB_SIZE:
            raise ValueError(f"model vocabulary ({config.vocab_size}) cannot represent ARC tokens ({ARC_VOCAB_SIZE})")
        model = CordForCausalLM.from_pretrained(args.resume_from) if args.resume_from else CordForCausalLM(config)
        parameter_count = unique_parameter_count(model)
        if parameter_count >= 50_000_000:
            raise RuntimeError(f"CoRD model has {parameter_count:,} unique parameters; ARC audit requires <50,000,000")
        device = torch.device(args.device)
        model.to(device)
        raw_groups = build_cord_optimizer_param_groups(model, weight_decay=args.weight_decay)
        optimizer = torch.optim.AdamW([
            {"params": group["params"], "weight_decay": group["weight_decay"]} for group in raw_groups
        ], lr=args.learning_rate)
        estimated_steps = max(1, (len(train_loader) * args.epochs + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps)
        scheduler = make_scheduler(optimizer, estimated_steps, args.warmup_ratio)
        metadata = {
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "parameter_count": parameter_count,
            "split_manifest_digest": split.manifest_digest,
            "train_tasks": len(split.train_files), "validation_tasks": len(split.validation_files), "test_tasks": len(test_files),
            "train_lengths": train_dataset.lengths(), "validation_lengths": validation_dataset.lengths(), "test_lengths": test_dataset.lengths(),
        }
        (output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        writer.add_text("run/config", json.dumps(metadata, indent=2), 0)
        writer.add_scalar("system/parameter_count", parameter_count, 0)
        print(json.dumps(metadata, indent=2))
        global_step = 0
        best_loss = float("inf")
        best_dir = output_dir / "best"
        if not args.eval_only:
            for epoch in range(args.epochs):
                train_dataset.set_epoch(epoch)
                global_step, train_metrics = train_epoch(
                    model, train_loader, optimizer, device, writer, epoch, global_step,
                    max_steps=args.smoke_optimizer_steps, gradient_accumulation_steps=args.gradient_accumulation_steps,
                    max_grad_norm=args.max_grad_norm, scheduler=scheduler,
                )
                validation_metrics = evaluate(model, validation_loader, device, writer, global_step, namespace="val")
                print(f"epoch={epoch} train={train_metrics} val={validation_metrics}")
                if validation_metrics["loss"] < best_loss:
                    best_loss = validation_metrics["loss"]
                    model.save_pretrained(best_dir)
                    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "global_step": global_step}, best_dir / "trainer_state.pt")
                if args.smoke_optimizer_steps is not None and global_step >= args.smoke_optimizer_steps:
                    break
        checkpoint_dir = args.resume_from if args.eval_only and args.resume_from else best_dir
        if checkpoint_dir is None or not checkpoint_dir.exists():
            raise FileNotFoundError("no best validation checkpoint exists; supply --resume-from for --eval-only")
        model = CordForCausalLM.from_pretrained(checkpoint_dir).to(device)
        test_metrics = evaluate(model, test_loader, device, writer, global_step, namespace="test")
        print(f"final_test={test_metrics}")
        (output_dir / "final_test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        writer.flush()
    finally:
        writer.close()


if __name__ == "__main__":
    main()
