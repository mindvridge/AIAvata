@echo off
chcp 65001 >nul
title DLL Fix - AI Avatar Service

echo.
echo ============================================================
echo        DLL 문제 해결 스크립트
echo ============================================================
echo.

cd /d "%~dp0"

echo [1/5] OpenCV 충돌 해결...
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y 2>nul
pip install opencv-python-headless --force-reinstall -q
echo       완료!

echo.
echo [2/5] MediaPipe 재설치...
pip uninstall mediapipe -y 2>nul
pip install mediapipe --force-reinstall -q
echo       완료!

echo.
echo [3/5] NumPy 호환성 확인...
pip install "numpy<2.0" --force-reinstall -q
echo       완료!

echo.
echo [4/5] protobuf 버전 맞춤...
pip install "protobuf>=3.20,<5.0" --force-reinstall -q
echo       완료!

echo.
echo [5/5] Visual C++ Redistributable 확인...
echo       Visual C++ Redistributable가 설치되어 있는지 확인하세요.
echo       없다면 다음 링크에서 다운로드하세요:
echo       https://aka.ms/vs/17/release/vc_redist.x64.exe
echo.

echo ============================================================
echo   DLL 문제 해결 완료!
echo   이제 run.bat를 다시 실행해주세요.
echo ============================================================
echo.

pause
