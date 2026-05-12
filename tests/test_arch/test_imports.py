# === INFERRED IMPORT CONTRACTS (review before committing) ===
#
# Subpackages (by dependency rank, highest -> lowest):
#   optimize_onnx (4) — fuses and rewrites graph nodes; imports pattern_detect, preprocess
#   model_validate (4) — orchestrates validation; imports optimize_onnx
#   structure_analysis (3) — topology analysis; imports preprocess
#   pattern_detect (2) — pure detection; no intra-package deps expected
#   preprocess (1) — version conversion, shape inference; utility leaf
#
# ALLOWED (not tested here):
#   optimize_onnx -> pattern_detect, optimize_onnx -> preprocess
#   model_validate -> optimize_onnx
#   structure_analysis -> preprocess
#
# FORBIDDEN (each becomes one test method below):
#   preprocess    -x-> optimize_onnx  [test_preprocess_does_not_import_optimize_onnx]
#   preprocess    -x-> model_validate [test_preprocess_does_not_import_model_validate]
#   pattern_detect -x-> optimize_onnx [test_pattern_detect_does_not_import_optimize_onnx]
#
# [REVIEW] Approve contract list before treating ARC3 as resolved.
# If a contract is wrong, delete the corresponding test method.
# ================================================================
"""Import architecture tests for slimonnx."""

__docformat__ = "restructuredtext"

import ast
import importlib
from pathlib import Path

import pytest

_SRC = Path(__file__).parent.parent.parent / "src" / "slimonnx"


def _get_imports(path: Path) -> set[str]:
    """Return top-level module names imported by path."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


class TestImportSmoke:
    """Package imports without circular dependencies or broken __init__ chains."""

    def test_top_level_import(self):
        """Import package top-level without error."""
        mod = importlib.import_module("slimonnx")
        assert mod.__name__ == "slimonnx"

    def test_submodule_imports_cleanly(self):
        """Every submodule imports without ImportError."""
        errors: list[str] = []
        for f in _SRC.rglob("*.py"):
            rel = f.relative_to(_SRC.parent).with_suffix("")
            mod = ".".join(rel.parts)
            try:
                importlib.import_module(mod)
            except ImportError as e:
                errors.append(f"{mod}: {e}")
        assert errors == [], "Import errors:\n" + "\n".join(errors)


class TestLayerBoundaries:
    """Inferred layer boundary tests — see contract header above."""

    @pytest.mark.parametrize(
        ("subpackage_dir", "forbidden_import"),
        [
            ("preprocess", "optimize_onnx"),
            ("preprocess", "model_validate"),
            ("pattern_detect", "optimize_onnx"),
        ],
    )
    def test_subpackage_does_not_import_forbidden(self, subpackage_dir: str, forbidden_import: str):
        """Lower-layer subpackages must not import higher-layer subpackages."""
        violations = [
            str(f.relative_to(_SRC))
            for f in (_SRC / subpackage_dir).rglob("*.py")
            if forbidden_import in _get_imports(f)
        ]
        assert violations == [], f"{subpackage_dir} imports {forbidden_import} in: {violations}"
