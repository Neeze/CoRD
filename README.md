# CoRD Transformers

Standalone Hugging Face Transformers project for the Concept-over-Recurrent-
Depth (CoRD) language model.

## Project layout

```text
cord/
├── configs/                 # Reproducible model configurations
├── docs/                    # Architecture and implementation notes
├── scripts/                 # Developer and profiling utilities
├── src/cord/                # Installable Python package
│   ├── configuration_cord.py
│   ├── modeling_cord.py
│   ├── modules.py
│   ├── state_graph.py
│   ├── cache_utils.py
│   ├── outputs.py
│   ├── training.py
│   └── __init__.py
├── tests/
│   ├── unit/
│   └── integration/
└── pyproject.toml
```

The first development target is the approximately 49M parameter prototype
described in `configs/cord-50m.json`. The 1B configuration from the research
blueprint is not used by default in tests.

Install the package into the workspace `astra` environment from this directory
with:

```bash
uv pip install --python ../astra/bin/python -e '.[test]'
```

The optional `kda` extra enables FLA kernels. The eager implementation remains
the portable reference backend for CPU tests and numerical parity checks.
