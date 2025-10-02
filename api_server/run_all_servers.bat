@echo off
echo ========================================
echo Starting All Face Recognition API Servers
echo ========================================
echo.
echo Starting servers on:
echo   - Vanilla CNN:   http://localhost:5001
echo   - ResNet50:      http://localhost:5002
echo   - Attention CNN: http://localhost:5003 (RECOMMENDED)
echo.
echo Press Ctrl+C in each window to stop
echo ========================================
echo.

start cmd /k "echo Vanilla CNN Server && python server_vanilla_cnn.py"
timeout /t 2 /nobreak >nul

start cmd /k "echo ResNet50 Server && python server_resnet50.py"
timeout /t 2 /nobreak >nul

start cmd /k "echo Attention CNN Server (RECOMMENDED) && python server_attention_cnn.py"

echo.
echo All servers started!
echo Check Swagger docs at:
echo   - http://localhost:5001/swagger/
echo   - http://localhost:5002/swagger/
echo   - http://localhost:5003/swagger/
pause

