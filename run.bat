@echo off
echo ========================================
echo Face Detection Model Comparison
echo ========================================
echo.
echo Starting training pipeline...
echo This will take approximately 30-60 minutes with GPU
echo or 2-4 hours with CPU.
echo.
echo Press Ctrl+C to cancel, or
pause

python main.py

echo.
echo ========================================
echo Training completed!
echo Check face_detection_results/ folder for outputs
echo ========================================
pause

