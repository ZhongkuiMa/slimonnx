# SlimONNX Architecture

ONNX model optimizer using flag-driven optimization passes, pattern detection, and numerical validation.

## Package Tree

```
src/slimonnx/
├── slimonnx.py            SlimONNX class entry point (modify when: changing pipeline orchestration)
├── configs.py             Frozen dataclasses: OptimizationConfig, ValidationConfig, AnalysisConfig
├── presets.py             Pre-tuned OptimizationConfig for VNN-COMP benchmarks
├── utils.py               Shared ONNX graph helpers (NOT domain logic)
├── preprocess/            Opset version conversion, graph cleanup (modify when: supporting new opsets)
│   ├── version_converter.py   Convert between ONNX opset versions
│   └── cleanup.py             Remove unused initializers, fix shapes
├── optimize_onnx/         Optimization passes (modify when: adding/changing fusions)
│   ├── _optimize.py       Pipeline orchestrator — dispatches to passes based on config flags
│   ├── _conv.py           Conv-BN fusion variants
│   ├── _bn_conv.py        BN-Conv fusion (reversed order)
│   ├── _depthwise_conv.py Depthwise conv fusion
│   ├── _gemm.py           MatMul+Add -> Gemm fusion
│   ├── _gemm_gemm.py      Consecutive Gemm fusion
│   ├── _bn_gemm.py        BN-Gemm fusion variants
│   ├── _bn_transpose.py   Transpose-BN-Transpose fusion
│   ├── _mm_add.py         MatMul+Add pattern
│   ├── _reshape.py        Conv-to-Flatten-Gemm simplification
│   ├── _cst_op.py         Constant folding
│   ├── _cst2initer.py     Constant nodes to initializers
│   ├── _redundant.py      Remove identity ops (add 0, mul 1, etc.)
│   ├── _dropout.py        Remove dropout nodes
│   ├── _ordering.py       Topological sort
│   ├── _name.py           Node name simplification
│   ├── _constants.py      Op type string constants
│   ├── _onnx_attrs.py     Attribute extraction helpers
│   └── _utils.py          Internal graph manipulation helpers
├── pattern_detect/        Pattern detection (modify when: detecting new fusible patterns)
│   ├── registry.py        Pattern registry + detect_all_patterns dispatcher
│   ├── conv_bn.py         Detect Conv-BN / BN-Conv patterns
│   ├── depthwise_conv.py  Detect depthwise conv patterns
│   ├── gemm_bn.py         Detect Gemm-BN variants
│   ├── gemm_chains.py     Detect consecutive Gemm
│   ├── matmul_add.py      Detect MatMul+Add
│   ├── redundant_ops.py   Detect identity operations
│   ├── reshape_chains.py  Detect consecutive reshapes
│   ├── reshape_negative_one.py  Detect problematic reshape dims
│   ├── constant_ops.py    Detect constant-foldable subgraphs
│   ├── dropout.py         Detect dropout nodes
│   ├── transpose_bn.py    Detect Transpose-BN-Transpose
│   └── utils.py           Detection helpers
├── model_validate/        Validation (modify when: adding validation checks)
│   ├── onnx_checker.py    ONNX spec compliance
│   ├── graph_validator.py Graph structural checks
│   ├── runtime_validator.py  OnnxRuntime inference checks
│   └── numerical_compare.py  Before/after output comparison
└── structure_analysis/    Analysis and reporting
    ├── analyzer.py        Model structure analysis
    ├── topology.py        Topology extraction
    └── reporter.py        JSON report generation
```

## Modification Map

| Intent | Primary Modify | Follow-ups | Avoid | Constraints | Failure Signal |
|--------|---------------|------------|-------|-------------|----------------|
| Add optimization pass | `optimize_onnx/_new.py` | `configs.py` (add flag), `_optimize.py` (wire flag), `pattern_detect/` (detection), tests | `slimonnx.py` (orchestration only) | Flag defaults to `False` (enforced) | `pytest tests/test_units/` fails |
| Add VNN-COMP preset | `presets.py` | `tests/test_benchmarks/` | `configs.py` (presets compose existing flags) | Alphabetical in `PRESET_NAMES` (observed) | Preset test fails |
| Add pattern detector | `pattern_detect/new.py` | `pattern_detect/registry.py`, `pattern_detect/__init__.py` | `optimize_onnx/` (detection is separate from execution) | Must register in `PATTERNS` (enforced) | `detect_all_patterns` misses pattern |
| Add validation check | `model_validate/` | `slimonnx.py` (wire into pipeline) | `optimize_onnx/` (validation is post-optimization) | - | `pytest tests/test_units/test_validation/` |
| Support new opset | `preprocess/version_converter.py` | `preprocess/cleanup.py` if needed | `optimize_onnx/` (opset-agnostic passes) | Opset 21 internal (enforced) | Shape inference fails |

## Dependency Rules

| Rule | Source | Failure |
|------|--------|---------|
| No relative imports | `ruff TID` (enforced) | `ruff check` error |
| `optimize_onnx/` must not import from `pattern_detect/` | (observed) | Circular concern mixing |
| `pattern_detect/` must not import from `optimize_onnx/` | (observed) | Circular concern mixing |
| `model_validate/` runs after optimization only | (enforced in `slimonnx.py`) | Validates wrong model |
| Internal opset is always 21 | (enforced in pipeline) | shapeonnx shape inference fails |

## Common Mistakes

| Mistake | Detection Signal | Fix |
|---------|-----------------|-----|
| Adding optimization flag but not wiring in `_optimize.py` | Flag has no effect; tests pass but optimization does nothing | Add dispatch in `_optimize.py` |
| Forgetting to re-export from `pattern_detect/__init__.py` | `from slimonnx.pattern_detect import detect_xxx` fails at import | Add to `__init__.py` and `__all__` |
| Mutating model in pattern detection | Detection changes graph state for later passes | Pattern detectors must be read-only; return match info only |

## Conventions

- Private modules in `optimize_onnx/` use `_` prefix; public API is `optimize_onnx.optimize_onnx()`
- Pattern detectors return lists of matched node groups (not modified graphs)
- All configs are frozen dataclasses (`@dataclass(frozen=True)`)
- `__docformat__ = "restructuredtext"` and `__all__` in every module

## Related Documents

- [README.md](README.md) -- usage and API
- [../ARCHITECTURE.md](../ARCHITECTURE.md) -- rover_alpha system architecture
