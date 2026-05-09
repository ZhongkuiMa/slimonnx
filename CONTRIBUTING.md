# Contributing to SlimONNX

Shared conventions (code style, type hints, docstrings, import rules) are in the
root [CONTRIBUTING.md](../CONTRIBUTING.md). This file covers slimonnx-specific workflow only.

## Setup

```bash
cd slimonnx
pip install -e ".[dev]"
pre-commit install
```

## Checks

```bash
pre-commit run --all-files   # lint, format, type-check
pytest tests/test_units/ -v  # tests
```

## Workflow

1. Create branch from `main`
2. Make changes
3. Run checks (above)
4. Commit and push

## Adding an Optimization Pass

The dominant contribution pattern is adding a new fusion/simplification optimization.

1. Create `src/slimonnx/optimize_onnx/_your_pass.py` with a function matching the signature of existing passes (takes `ModelProto`, returns `ModelProto`)
2. Add a corresponding flag to `src/slimonnx/configs.py` in `OptimizationConfig` (default `False`)
3. Wire the flag in `src/slimonnx/optimize_onnx/_optimize.py`
4. Add pattern detection in `src/slimonnx/pattern_detect/` (new module + register in `registry.py`)
5. Re-export from `src/slimonnx/pattern_detect/__init__.py`
6. Add unit tests in `tests/test_units/test_optimize/`
7. Run: `pytest tests/test_units/ -v`

## Adding a Preset

1. Add the config to `src/slimonnx/presets.py` (alphabetical order in `PRESET_NAMES`)
2. Add benchmark test in `tests/test_benchmarks/`

## Constraints

| Rule | Details |
|------|---------|
| Absolute imports only | `from slimonnx.xxx import yyy` (no relative imports, enforced by ruff TID) |
| Frozen dataclasses | All configs use `@dataclass(frozen=True)` |
| `__docformat__` + `__all__` | Required in every module |
| Private modules | Optimization passes use `_` prefix; public API via `__init__.py` re-exports |
| Always-on transforms | `constant_to_initializer`, `simplify_gemm`, `reorder_by_strict_topological_order` always run |
