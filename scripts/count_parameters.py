"""Print the configuration used for a CoRD parameter-count run."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cord import CordConfig, CordForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = CordConfig.from_json_file(args.config)
    model = CordForCausalLM(config)
    unique_parameters = {id(parameter): parameter for parameter in model.parameters()}
    total = sum(parameter.numel() for parameter in unique_parameters.values())
    print(f"parameters={total}")
    print(f"parameters_millions={total / 1_000_000:.3f}")


if __name__ == "__main__":
    main()
