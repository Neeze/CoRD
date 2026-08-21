import argparse
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "train_arc.py"
SPEC = importlib.util.spec_from_file_location("cord_train_arc_cli", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parallel_cli_accepts_device_list_and_per_device_batch_size():
    args = MODULE.parse_args([
        "--strategy", "ddp", "--devices", "0", "1", "--batch-size", "2"
    ])
    MODULE.validate_parallel_args(args)
    assert args.strategy == "ddp"
    assert args.devices == [0, 1]
    assert args.batch_size == 2


@pytest.mark.parametrize("strategy", ["fsdp", "deepspeed_stage_3"])
def test_unimplemented_sharded_strategies_fail_explicitly(strategy):
    args = argparse.Namespace(
        strategy=strategy, devices=[0, 1], batch_size=1
    )
    with pytest.raises(NotImplementedError, match="not implemented"):
        MODULE.validate_parallel_args(args)


def test_ddp_requires_multiple_unique_devices():
    args = MODULE.parse_args([
        "--strategy", "ddp", "--devices", "0", "--batch-size", "2"
    ])
    with pytest.raises(ValueError, match="at least two"):
        MODULE.validate_parallel_args(args)

    args.devices = [0, 0]
    with pytest.raises(ValueError, match="unique"):
        MODULE.validate_parallel_args(args)


def test_plain_ddp_invocation_relaunches_through_torchrun(monkeypatch):
    args = MODULE.parse_args([
        "--strategy", "ddp", "--devices", "0", "1", "--batch-size", "2"
    ])
    captured = {}

    def run(command, *, check, env):
        captured.update(command=command, check=check, env=env)

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(MODULE.subprocess, "run", run)
    monkeypatch.setattr(MODULE.sys, "argv", [str(SCRIPT), "--strategy", "ddp", "--devices", "0", "1"])
    assert MODULE.launch_distributed_if_needed(args)
    assert captured["check"] is True
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "torch.distributed.run" in captured["command"]
    assert captured["command"][captured["command"].index("--nproc_per_node") + 1] == "2"


def test_training_plan_reports_all_segments_and_token_budgets():
    plan = MODULE.build_training_plan(
        {"samples": 10, "total_tokens": 1000, "prefix_tokens": 800, "supervised_tokens": 200},
        sft_epochs=50,
        awr_epochs=2,
        ppo_epochs=1,
        steps_per_sft_epoch=3,
        steps_per_graph_epoch=5,
        verified_leaves=4,
        decode_max_new_tokens=100,
    )
    assert [segment["name"] for segment in plan["segments"]] == ["SFT", "AWR", "PPO"]
    assert plan["scheduled_tokens"]["input"] == 53_000
    assert plan["scheduled_tokens"]["supervised"] == 10_600
    assert plan["scheduled_tokens"]["graph_decode_upper_bound"] == 12_000
    assert plan["total_optimizer_steps"] == 165
