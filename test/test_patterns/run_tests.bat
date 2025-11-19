@echo off
set "PYTHONPATH=C:\Users\zhong\Sync\Codes\GitHub\rover_alpha;%PYTHONPATH%"
cd /d "C:\Users\zhong\Sync\Codes\GitHub\rover_alpha\slimonnx\test\test_patterns"

echo Creating test pattern models...
python create_patterns.py
if errorlevel 1 (
    echo FAILED: Pattern creation failed
    exit /b 1
)

echo.
echo Running pattern detection tests...
python test_pattern_detect.py
if errorlevel 1 (
    echo FAILED: Pattern detection tests failed
    exit /b 1
)

echo.
echo Running optimization tests...
python test_optimize.py
if errorlevel 1 (
    echo FAILED: Optimization tests failed
    exit /b 1
)

echo.
echo Running optimize_onnx module tests...
python test_optimize_onnx.py
if errorlevel 1 (
    echo FAILED: optimize_onnx tests failed
    exit /b 1
)

echo.
echo SUCCESS: All tests passed
