import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cord import CordConfig, CordForCausalLM


def test_prototype_parameter_budget():
    config_path = Path(__file__).parents[2] / "configs" / "cord-50m.json"
    config = CordConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    model = CordForCausalLM(config)
    unique_parameters = {id(parameter): parameter for parameter in model.parameters()}
    parameter_count = sum(parameter.numel() for parameter in unique_parameters.values())
    assert parameter_count < 50_000_000
