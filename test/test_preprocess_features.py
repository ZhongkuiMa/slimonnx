"""Test enhanced preprocessing features: version conversion, docstring clearing, marking."""

__docformat__ = "restructuredtext"
__all__ = [
    "test_version_conversion",
    "test_docstring_clearing",
    "test_slimonnx_marking",
]

import os
import sys
import warnings

from slimonnx import SlimONNX, RECOMMENDED_OPSET, MAX_TESTED_OPSET


def test_version_conversion():
    """Test ONNX version conversion with warning system."""
    print("\n" + "=" * 80)
    print("TEST: Version Conversion")
    print("=" * 80)

    test_model = "benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx"
    if not os.path.exists(test_model):
        print(f"Test model not found: {test_model}")
        return False

    slimonnx = SlimONNX()

    # Test 1: Convert to recommended version
    print(f"\n1. Convert to recommended opset {RECOMMENDED_OPSET}")
    model = slimonnx.preprocess(test_model, target_opset=RECOMMENDED_OPSET)
    current_opset = model.opset_import[0].version if model.opset_import else 0
    print(f"   Result: opset={current_opset}, IR={model.ir_version}")
    assert (
        current_opset == RECOMMENDED_OPSET
    ), f"Expected opset {RECOMMENDED_OPSET}, got {current_opset}"

    # Test 2: Convert to max tested version
    print(f"\n2. Convert to max tested opset {MAX_TESTED_OPSET}")
    model = slimonnx.preprocess(test_model, target_opset=MAX_TESTED_OPSET)
    current_opset = model.opset_import[0].version if model.opset_import else 0
    print(f"   Result: opset={current_opset}, IR={model.ir_version}")
    assert (
        current_opset == MAX_TESTED_OPSET
    ), f"Expected opset {MAX_TESTED_OPSET}, got {current_opset}"

    # Test 3: Convert to outside range (should warn or fail)
    print("\n3. Convert to opset 22 (outside tested range, should warn/fail)")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = slimonnx.preprocess(test_model, target_opset=22)
            if len(w) > 0:
                print(f"   Warning: {w[0].message}")
            else:
                print("   No warning issued")
        current_opset = model.opset_import[0].version if model.opset_import else 0
        print(f"   Result: opset={current_opset}, IR={model.ir_version}")
    except Exception as e:
        print(f"   Expected failure: opset 22 is not supported by ONNX")
        print(f"   Error: {str(e)[:100]}")

    # Test 4: No conversion (keep original)
    print("\n4. Keep original version (target_opset=None)")
    model = slimonnx.preprocess(test_model, target_opset=None)
    current_opset = model.opset_import[0].version if model.opset_import else 0
    print(f"   Result: opset={current_opset}, IR={model.ir_version}")

    print("\nOK: Version conversion tests passed")
    return True


def test_docstring_clearing():
    """Test docstring clearing functionality."""
    print("\n" + "=" * 80)
    print("TEST: Docstring Clearing")
    print("=" * 80)

    test_model = "benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx"
    if not os.path.exists(test_model):
        print(f"Test model not found: {test_model}")
        return False

    slimonnx = SlimONNX()

    # Test 1: Without clearing
    print("\n1. Load without clearing docstrings")
    model = slimonnx.preprocess(test_model, clear_docstrings=False)
    docstrings_present = sum(1 for node in model.graph.node if node.doc_string)
    print(f"   Nodes with docstrings: {docstrings_present}/{len(model.graph.node)}")

    # Test 2: With clearing
    print("\n2. Load with clearing docstrings")
    model = slimonnx.preprocess(test_model, clear_docstrings=True)
    docstrings_present = sum(1 for node in model.graph.node if node.doc_string)
    print(f"   Nodes with docstrings: {docstrings_present}/{len(model.graph.node)}")
    assert docstrings_present == 0, f"Expected 0 docstrings, got {docstrings_present}"

    print("\nOK: Docstring clearing tests passed")
    return True


def test_slimonnx_marking():
    """Test SlimONNX marking functionality."""
    print("\n" + "=" * 80)
    print("TEST: SlimONNX Marking")
    print("=" * 80)

    test_model = "benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx"
    if not os.path.exists(test_model):
        print(f"Test model not found: {test_model}")
        return False

    slimonnx = SlimONNX()

    # Test 1: Without marking
    print("\n1. Load without SlimONNX marking")
    model = slimonnx.preprocess(test_model, mark_slimonnx=False)
    print(f"   Producer: {model.producer_name}")
    print(f"   Doc string: {model.doc_string}")

    # Test 2: With marking
    print("\n2. Load with SlimONNX marking")
    model = slimonnx.preprocess(test_model, mark_slimonnx=True)
    print(f"   Producer: {model.producer_name}")
    print(f"   Doc string: {model.doc_string}")
    assert "SlimONNX" in model.producer_name, "Producer name not marked"
    assert "SlimONNX" in model.doc_string, "Doc string not marked"

    print("\nOK: SlimONNX marking tests passed")
    return True


def test_all_features():
    """Test all preprocessing features together."""
    print("\n" + "=" * 80)
    print("TEST: Combined Features")
    print("=" * 80)

    test_model = "benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx"
    if not os.path.exists(test_model):
        print(f"Test model not found: {test_model}")
        return False

    slimonnx = SlimONNX()

    # Test all features enabled
    print(f"\nPreprocess with all features enabled:")
    print(
        f"  - target_opset={RECOMMENDED_OPSET} (recommended: {RECOMMENDED_OPSET}-{MAX_TESTED_OPSET})"
    )
    print("  - infer_shapes=True")
    print("  - clear_docstrings=True")
    print("  - mark_slimonnx=True")

    model = slimonnx.preprocess(
        test_model,
        target_opset=RECOMMENDED_OPSET,
        infer_shapes=True,
        clear_docstrings=True,
        mark_slimonnx=True,
    )

    current_opset = model.opset_import[0].version if model.opset_import else 0
    docstrings_present = sum(1 for node in model.graph.node if node.doc_string)

    print(f"\nResults:")
    print(f"  Opset version: {current_opset}")
    print(f"  IR version: {model.ir_version}")
    print(f"  Docstrings cleared: {docstrings_present == 0}")
    print(f"  Producer marked: {'SlimONNX' in model.producer_name}")
    print(f"  Doc string marked: {'SlimONNX' in model.doc_string}")

    assert current_opset == RECOMMENDED_OPSET
    assert docstrings_present == 0
    assert "SlimONNX" in model.producer_name
    assert "SlimONNX" in model.doc_string

    print("\nOK: Combined features test passed")
    return True


if __name__ == "__main__":
    success = True

    try:
        success &= test_version_conversion()
    except Exception as e:
        print(f"\nFAILED: Version conversion test: {e}")
        success = False

    try:
        success &= test_docstring_clearing()
    except Exception as e:
        print(f"\nFAILED: Docstring clearing test: {e}")
        success = False

    try:
        success &= test_slimonnx_marking()
    except Exception as e:
        print(f"\nFAILED: SlimONNX marking test: {e}")
        success = False

    try:
        success &= test_all_features()
    except Exception as e:
        print(f"\nFAILED: Combined features test: {e}")
        success = False

    print("\n" + "=" * 80)
    if success:
        print("SUCCESS: All preprocessing feature tests passed")
    else:
        print("FAILED: Some preprocessing feature tests failed")
    print("=" * 80)

    sys.exit(0 if success else 1)
