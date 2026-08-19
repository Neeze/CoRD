"""Compare direct greedy ARC decoding with detached CoRD graph search."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cord import CordForCausalLM, CordSearchConfig
from dataset.data import ARCDataset, collate_fn, discover_arc_tasks, split_arc_training_files
from trainer.arc_search_evaluation import compare_direct_and_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "datasets" / "ARC-AGI-2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-expansions", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=2)
    parser.add_argument("--max-verified-leaves", type=int, default=2)
    parser.add_argument("--decode-max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    split = split_arc_training_files(discover_arc_tasks(args.data_dir / "training"), seed=args.seed)
    dataset = ARCDataset(split.validation_files, augment=False, augmentation_seed=args.seed, split_name="validation")
    dataloader = DataLoader(dataset, args.batch_size, shuffle=False, collate_fn=partial(collate_fn, max_length=None))
    device = torch.device(args.device)
    model = CordForCausalLM.from_pretrained(args.checkpoint).to(device)
    metrics = compare_direct_and_graph(
        model,
        dataloader,
        device,
        search_config=CordSearchConfig(
            max_expansions=args.max_expansions,
            beam_size=args.beam_size,
            max_verified_leaves=args.max_verified_leaves,
            deterministic=True,
            seed=args.seed,
        ),
        max_steps=args.max_steps,
        decode_max_new_tokens=args.decode_max_new_tokens,
    )
    report = {
        "split": "training-validation (never ARC evaluation)",
        "split_manifest_digest": split.manifest_digest,
        "search_config": {
            "max_expansions": args.max_expansions,
            "beam_size": args.beam_size,
            "max_verified_leaves": args.max_verified_leaves,
            "seed": args.seed,
            "decode_max_new_tokens": args.decode_max_new_tokens,
        },
        "metrics": metrics,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
