# Slimonnx Conventions

This file defines style and documentation conventions for the slimonnx package.
Use it as a **checklist** — when writing or reviewing code, check each item below
one by one.

---

## 1. Module Docstrings

Every `.py` file begins with a module docstring.

### Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 1.1 | **First line**: short summary of the module's purpose (one sentence) | ☐ |
| 1.2 | **Extended description** (optional): 1-2 paragraphs after a blank line, covering the module's role in the optimization pipeline | ☐ |
| 1.3 | **Format**: ReST plain text, no `:param:` or `:return:` tags at module level | ☐ |
| 1.4 | Always followed by `__docformat__ = "restructuredtext"` | ☐ |
| 1.5 | **No author, date, or version lines** — git history is authoritative | ☐ |
| 1.6 | **No non-ASCII characters** in docstrings — use ASCII equivalents | ☐ |

### Patterns

| File type | Style | Example |
|-----------|-------|---------|
| Operation module (`_bn_conv.py`, `_gemm.py`) | One line | `"""Fuse BatchNormalization with Conv and ConvTranspose operators."""` |
| Pipeline module (`_optimize.py`) | One line | `"""Main optimization orchestration for ONNX models."""` |
| Registry module (`registry.py`) | One line | `"""Pattern detection registry."""` |
| Utility module (`utils.py`) | One line | `"""Utility functions for ONNX model manipulation and analysis."""` |
| Constants module (`constants.py`) | Summary + paragraph | `"""Package-level constants... Holds ONNX-spec defaults..."""` |
| Entry point (`slimonnx.py`) | One line | `"""SlimONNX: ONNX model optimization and analysis toolkit."""` |
| `__init__.py` | One line describing the subpackage | `"""Pattern detection for ONNX model optimization."""` |

---

## 2. Function/Class Docstrings

### 2.1 Structure

```python
def func_name(param1: type, param2: type) -> return_type:
    """
    Short imperative description of what the function computes.

    Extended description (optional) — the algorithm or pipeline step.

    :param param1: Description of param1 (capitalized, ends with period).
    :param param2: Description of param2.

    :return: Description of return value(s) (capitalized, ends with period).
    :raises ValueError: When and why this exception is raised.
    """
```

### 2.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 2.1 | **First line**: imperative mood, describes what the function computes, ends with period | ☐ |
| 2.2 | Use `:param name:`, `:return:`, and `:raises ExceptionType:` tags — no `:type:` tags | ☐ |
| 2.3 | `:param` descriptions: **capitalized, end with period**, describe semantics not types | ☐ |
| 2.4 | `:return` description: **capitalized, end with period** | ☐ |
| 2.5 | Private helpers (`_apply_*`, `_check_*`) may use a single-line docstring without `:param:` tags | ☐ |
| 2.6 | Public API functions (`optimize_onnx`, `slim`, `detect_all_patterns`) require full `:param:` documentation for every parameter | ☐ |
| 2.7 | No docstring on `__init__` of a dataclass (the class docstring covers it) | ☐ |
| 2.8 | **No non-ASCII characters** in docstrings | ☐ |
| 2.9 | **No bold-header sections** in function docstrings — no `**Example**:`, `**Note**:`; body contains only description prose, `:param:`, `:return:`, `:raises:` | ☐ |

### 2.3 Good examples

```python
def slim(
    self,
    onnx_path: str,
    target_path: str | None = None,
    config: OptimizationConfig | None = None,
    validation: bool = False,
) -> dict | None:
    """
    Optimize ONNX model.

    The optimization pipeline:
    1. Load model
    2. Convert to Opset 21 for compatibility with shapeonnx
    3. Apply optimizations based on config
    4. Validate output if requested
    5. Save optimized model

    :param onnx_path: Path to input ONNX model.
    :param target_path: Path to save optimized model (default: {input}_simplified.onnx).
    :param config: Optimization configuration (default: OptimizationConfig()).
    :param validation: Whether to run validation comparing outputs.
    :return: Optimization report if validation enabled, else None.
    """
```

```python
def clear_onnx_docstring(model: ModelProto) -> ModelProto:
    """Remove all doc_string entries from nodes in the ONNX model.

    :param model: ONNX model to clean.
    :return: Model with all doc_strings cleared.
    """
```

---

## 3. Inline Comments

| # | Rule | Pass/Fail |
|---|------|-----------|
| 3.1 | Comment **why**, not what — the code already says what | ☐ |
| 3.2 | Only add comments when the reasoning is non-obvious (algorithm rationale, ordering constraints) | ☐ |
| 3.3 | **No inline shape comments** on function signatures — shapes belong in `:param:`/`:return:` docstrings | ☐ |
| 3.4 | No commented-out code — delete it | ☐ |
| 3.5 | `# TODO:` comments require an associated issue reference (enforced by ruff TD001) | ☐ |
| 3.6 | Section comments in large functions: `# Step 1: ...`, `# Step 2: ...` for numbered pipeline stages | ☐ |

---

## 4. Naming Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 4.1 | **Classes**: PascalCase — `SlimONNX`, `OptimizationConfig`, `ValidationConfig` | ☐ |
| 4.2 | **Functions/methods**: snake_case — `optimize_onnx`, `detect_all_patterns`, `clear_onnx_docstring` | ☐ |
| 4.3 | **Private functions**: `_` prefix — `_apply_conv_fusions`, `_fuse_conv_bn_or_bn_conv` | ☐ |
| 4.4 | **Private modules**: `_` prefix — `_optimize.py`, `_bn_conv.py`, `_gemm.py`. Exception: `__init__.py` | ☐ |
| 4.5 | **Constants**: UPPER_CASE — `DEFAULT_GEMM_ALPHA`, `DEFAULT_GEMM_BETA`, `PATTERNS` | ☐ |
| 4.6 | **Boolean parameters**: verb phrases — `fuse_conv_bn`, `remove_dropout`, `has_batch_dim` (no `if_` prefix) | ☐ |
| 4.7 | **ONNX protobuf names**: `NodeProto`, `TensorProto`, `ModelProto`, `ValueInfoProto` — match ONNX API exactly | ☐ |
| 4.8 | **Detection function prefix**: `detect_` for pattern detectors — `detect_conv_bn`, `detect_dropout`, `detect_matmul_add` | ☐ |

---

## 5. Code Style

| # | Rule | Pass/Fail |
|---|------|-----------|
| 5.1 | **100-char line length** (enforced by ruff) | ☐ |
| 5.2 | **Double quotes** for strings and docstrings | ☐ |
| 5.3 | **Absolute imports only** — `from slimonnx.utils import get_initializers` | ☐ |
| 5.4 | `__docformat__ = "restructuredtext"` after module docstring, before imports | ☐ |
| 5.5 | `__all__` in every source module, alphabetically sorted, listing all public names. Not required in test files. | ☐ |
| 5.6 | **Import order**: stdlib → third-party (`onnx`, `numpy`) → first-party (`slimonnx.*`). Groups separated by blank lines. | ☐ |
| 5.7 | `import numpy as np`; `from onnx import NodeProto, TensorProto` (for type annotations) | ☐ |
| 5.8 | `logger = logging.getLogger(__name__)` at module level in entry-point modules | ☐ |
| 5.9 | **McCabe complexity ≤ 10** (enforced by ruff C90) | ☐ |
| 5.10 | **Only import what you use** — clean up unused imports (enforced by ruff F401) | ☐ |
| 5.11 | **No string annotations** when type is already imported — write `-> ModelProto` not `-> "ModelProto"` | ☐ |
| 5.12 | **No imports inside function bodies** (E402). Exception: lazy imports inside `SlimONNX` methods to avoid circular imports between subpackages. | ☐ |

---

## 6. ONNX Operation Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 6.1 | **Node list signature**: optimization functions accept `nodes: list[NodeProto]` and return `list[NodeProto]` | ☐ |
| 6.2 | **Initializer dict**: `initializers: dict[str, TensorProto]` passed alongside nodes for weight access | ☐ |
| 6.3 | **In-place modification**: functions modify `nodes` and `initializers` in place where safe; document mutation in docstring | ☐ |
| 6.4 | **Shape inference**: use `from shapeonnx.infer_shape import infer_onnx_shape` for shape queries during optimization | ☐ |
| 6.5 | **Node matching**: iterate `nodes` by index, match patterns by op_type (`node.op_type == "Conv"`) and connectivity | ☐ |
| 6.6 | **New node creation**: use `onnx.NodeProto()` with `CopyFrom` + `ClearField` pattern for constructing fused nodes from existing node attributes | ☐ |
| 6.7 | **Pattern detection return**: detectors return `list[dict]` where each dict describes one matched pattern instance | ☐ |

---

## 7. Optimization Pipeline Patterns

| # | Rule | Pass/Fail |
|---|------|-----------|
| 7.1 | Pipeline steps are numbered in `slim()` docstring and code comments | ☐ |
| 7.2 | Each optimization domain (conv, gemm, reshape, etc.) has its own private module `_<domain>.py` in `optimize_onnx/` | ☐ |
| 7.3 | Fusion functions are grouped by domain: `_apply_conv_fusions()`, `_apply_gemm_fusions()`, `_apply_shape_optimizations()` | ☐ |
| 7.4 | Each domain's fusion function takes boolean flags controlling individual optimizations | ☐ |
| 7.5 | Shape inference is called once per optimization group, not per individual fusion | ☐ |
| 7.6 | `has_batch_dim: bool` is threaded through all optimization functions to handle batch-aware vs batch-agnostic shapes | ☐ |
| 7.7 | Optimizations that depend on shape inference run after shape-based passes | ☐ |

---

## 8. Pattern Detect Registry Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 8.1 | `PATTERNS` dict at module level in `registry.py` maps pattern name → `{"description", "category", "severity"}` | ☐ |
| 8.2 | Pattern categories: `"fusion"`, `"redundant"`, `"inference"`, `"constant_folding"`, `"shape_optimization"` | ☐ |
| 8.3 | Severity levels: `"optimization"`, `"info"`, `"redundant"` | ☐ |
| 8.4 | `_build_detector_registry()` constructs the mapping from pattern names to detector functions | ☐ |
| 8.5 | Each detector module exports one or more `detect_<pattern>` functions. Modules are registered via `DetectorSig` enum. Some modules export multiple detectors (e.g., `conv_bn.py` exports 4, `redundant_ops.py` exports 6) | ☐ |
| 8.6 | `detect_all_patterns()` is the single public entry point for pattern detection | ☐ |
| 8.7 | Pattern keys use snake_case strings — `"matmul_add"`, `"conv_bn"`, `"add_zero"` | ☐ |

---

## 9. Model Validate Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 9.1 | `_orchestrator.py` coordinates validation: check → compare → report | ☐ |
| 9.2 | Validator modules may export multiple public functions (e.g., `graph_validator.py` exports 5, `numerical_compare.py` exports 3). `_orchestrator.py` coordinates calling them | ☐ |
| 9.3 | `onnx_checker.py` wraps `onnx.checker.check_model()` | ☐ |
| 9.4 | `numerical_compare.py` compares outputs with `np.allclose(rtol=..., atol=...)`; uses `onnxruntime.InferenceSession` for inference | ☐ |
| 9.5 | `DetectorSig.NS` and `NS_INIT` detectors silently skip when `data_shapes` is `None` — this degradation path is intentional | ☐ |

---

## 10. Frozen Dataclass Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 10.1 | Configuration classes use `@dataclass(frozen=True)` (e.g., `OptimizationConfig`, `ValidationConfig`) | ☐ |
| 10.2 | Every field has an explicit type annotation | ☐ |
| 10.3 | Default values are simple types or `None` (not mutable containers) | ☐ |
| 10.4 | Class docstring explains the config's purpose; `:param name:` for each field | ☐ |

---

## 11. Architecture Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 11.1 | **`__init__.py` facade pattern**: import from private `_*.py` modules, re-export via `__all__` | ☐ |
| 11.2 | **Subpackage dependency flow**: `preprocess → optimize_onnx → model_validate`. pattern_detect and structure_analysis are independent. | ☐ |
| 11.3 | **Root modules** (`constants.py`, `utils.py`, `configs.py`, `presets.py`) must NOT import from subpackages | ☐ |
| 11.4 | **Subpackages import from root modules** — `from slimonnx.utils import get_initializers` | ☐ |
| 11.5 | **shapeonnx is an optional dependency**: import guarded or at module level (always available in dev installs) | ☐ |
| 11.6 | Constants shared across subpackages go in root `constants.py`; subpackage-internal constants go in the subpackage's `_constants.py` or inline in the relevant module if used in only one file | ☐ |
| 11.7 | `has_constant_weight(initializers, weight_index=1)` assumes weight is always the second input (input[0]=data, input[1]=weight) per ONNX convention | ☐ |
| 11.8 | `data_shapes` type is `dict[str, int | list[int]] | None` at the registry level; individual detectors may use stricter `dict[str, list[int]] | None` | ☐ |

---

## 12. Constants Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 12.1 | **Naming**: `UPPER_SNAKE_CASE`, 2-4 words. Use prefixes (`DEFAULT_`, `MAX_`, `MIN_`) and suffixes (`_DIR`, `_NAME`, `_MB`) for clarity | ☐ |
| 12.2 | **Scope levels**: Place at narrowest scope — function-level → file-level → subfolder `_constants.py` → package-level `constants.py`. Promote when a second consumer at broader scope appears | ☐ |
| 12.3 | **Extraction trigger**: Extract a literal when it appears 2+ times. Never duplicate a constant across files | ☐ |
| 12.4 | **When NOT to extract**: Self-documenting single-use values, test data, function defaults already named by the parameter, `0`/`1`/`-1` for indexing | ☐ |
| 12.5 | **Type annotations**: Annotate only when the type is not obvious from the literal | ☐ |
| 12.6 | **Frozen collections**: Use `frozenset` or `tuple` for constant collections — never mutable `list` or `set` | ☐ |
| 12.7 | **Shared vs subpackage constants**: Shared across subpackages → root `constants.py`; subpackage-internal → subpackage `_constants.py` or inline if used in one file | ☐ |

---

## 13. Enum Conventions

| # | Rule | Pass/Fail |
|---|------|-----------|
| 13.1 | **IntEnum with `@unique`**: All enums use `IntEnum` with `@unique` decorator | ☐ |
| 13.2 | **Placement**: Subpackage-local enums in `<subfolder>/_enums.py` (e.g., `pattern_detect/_enums.py`) | ☐ |
| 13.3 | **Class naming**: PascalCase with categorical suffix — `Mode`, `Type`, `Status`, `Strategy`. Never suffix with `Enum` | ☐ |
| 13.4 | **Member naming**: `UPPER_SNAKE_CASE`, 1-3 words | ☐ |
| 13.5 | **Custom `__repr__`**: IntEnum classes define `__repr__` returning `f"{self.name}"` | ☐ |
| 13.6 | **Member docstrings**: Every enum member has a one-line ReST docstring after the value assignment | ☐ |
| 13.7 | **Module boilerplate**: `__docformat__ = "restructuredtext"`, `__all__` alphabetically sorted | ☐ |

---

## 14. Test Style

### 14.1 Directory Layout

```
tests/
├── test_arch/                 # architecture/import enforcement
├── test_benchmarks/           # integration tests (opt-in)
│   └── baselines/test/
├── test_units/
│   ├── test_analyze_structure/
│   ├── test_model_validate/
│   ├── test_optimize_onnx/
│   ├── test_pattern_detect/
│   ├── test_preprocess/
│   ├── test_presets/
│   ├── test_slimonnx/
│   ├── test_structure_analysis/
│   └── test_utils/
```

### 14.2 Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 14.1 | **Test file naming**: `test_<concern>.py` — `test_bn_conv.py`, `test_patterns.py` | ☐ |
| 14.2 | **One test folder per source subpackage**: `test_optimize_onnx/` ↔ `optimize_onnx/` | ☐ |
| 14.3 | `__init__.py` at leaf `test_<pkg>/` level only (collision avoidance) | ☐ |
| 14.4 | `_helpers.py` for shared test utilities within a test subpackage | ☐ |
| 14.5 | **No pytest markers** except `@pytest.mark.parametrize` | ☐ |
| 14.6 | Test module docstrings: 1-3 lines max summarizing what the file validates | ☐ |
| 14.7 | Benchmark tests go in `test_benchmarks/` and are excluded from default `pytest` runs | ☐ |
| 14.8 | ONNX test models are generated in fixtures or loaded from test data files | ☐ |
| 14.9 | **Default test suite**: `pytest` runs `tests/test_units/` by default. Benchmark tests in `test_benchmarks/` are opt-in | ☐ |
| 14.10 | **No `@pytest.mark.skip`** in committed code — use conditional early return with `[REVIEW]` comment | ☐ |

---

## 15. Logging Conventions

Pipeline tool: use `logging` package with `_enable_verbose()` helper.

### Setup

```python
import logging

_logger = logging.getLogger(__name__)


def _enable_verbose() -> None:
    """Configure package-level logger for console output."""
    pkg_logger = logging.getLogger("slimonnx")
    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        pkg_logger.addHandler(handler)
    pkg_logger.setLevel(logging.DEBUG)


class SlimONNX:
    def __init__(self, verbose: bool = False):
        if verbose:
            _enable_verbose()
```

### Rules

| # | Rule | Pass/Fail |
|---|------|-----------|
| 15.1 | **`_enable_verbose()` in `slimonnx.py`** — single configuration point for the package | ☐ |
| 15.2 | **Direct `_logger.info(f"...")` calls** — no `isEnabledFor` guards, no `%`-formatting | ☐ |
| 15.3 | **f-strings for all log messages** — `_logger.info(f"  Preprocess: loaded (0.0123s)")` | ☐ |
| 15.4 | **Output format**: first line `SlimONNX: optimizing <path>`, stage lines `  Stage: description (0.XXXXs)`, final line `  Total: 0.XXXXs` | ☐ |
| 15.5 | **Per-stage timing**: `t = time.perf_counter()` before each stage, elapsed after | ☐ |
| 15.6 | **`warnings.warn()` for recoverable errors** — never `logger.warning()`. Warnings fire regardless of verbose flag. Use `stacklevel=2` | ☐ |
| 15.7 | **`raise ValueError/RuntimeError` for fatal errors** — never `logger.error()` | ☐ |
| 15.8 | **No `print()` for diagnostic output** in source code | ☐ |
| 15.9 | **Sub-module warnings**: files like `analyze_structure.py`, `numerical_compare.py`, `runtime_validator.py` use `warnings.warn()` only, no logger | ☐ |
