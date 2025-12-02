@echo off
chcp 65001 >nul
title AI Avatar Service

echo.
echo ============================================================
echo            AI Avatar Service - Windows
echo ============================================================
echo.

cd /d "%~dp0"

:: DLL 오류 발생 시 fix_dll.bat 실행 안내
if "%1"=="--fix" goto :fix_dll

echo [1/4] 패키지 확인 중...
python -c "import torch,openai,livekit" 2>nul
if errorlevel 1 (
    echo       패키지 설치 중...
    pip install torch torchaudio torchvision -q
    pip install fastapi uvicorn python-dotenv websockets aiofiles pydantic -q
    pip install "numpy<2.0" -q
    pip install opencv-python-headless -q
    pip install openai anthropic livekit livekit-api -q
    pip install transformers diffusers huggingface_hub -q
    pip install funasr modelscope omegaconf kaldiio -q
    pip install edge-tts gtts chatterbox-tts resemble-perth -q
    pip install mediapipe "protobuf>=3.20,<5.0" -q
    echo       완료!
) else (
    echo       이미 설치됨 - 스킵
)

echo.
echo [2/4] 모델 확인 중...
if not exist "models\musetalk\sd-vae-ft-mse\config.json" (
    echo       모델 다운로드 중...
    python -c "from huggingface_hub import snapshot_download; from pathlib import Path; Path('models/musetalk').mkdir(parents=True,exist_ok=True); snapshot_download('stabilityai/sd-vae-ft-mse',local_dir='models/musetalk/sd-vae-ft-mse')"
    echo       완료!
) else (
    echo       이미 존재 - 스킵
)

echo.
echo [3/4] 환경 설정 확인 중...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo       .env 파일 생성됨
        echo       [!] API 키를 설정해주세요!
    )
) else (
    echo       .env 파일 존재
)

echo.
echo [4/4] 서버 시작...
echo.
echo ============================================================
echo   서버 주소: http://localhost:8000
echo   API 문서:  http://localhost:8000/docs
echo   종료: Ctrl+C
echo.
echo   [!] DLL 오류 발생 시: run.bat --fix 또는 fix_dll.bat 실행
echo ============================================================
echo.

python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
goto :end

:fix_dll
echo.
echo [DLL 문제 해결 모드]
echo.
echo [1/4] OpenCV 충돌 해결...
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y 2>nul
pip install opencv-python-headless --force-reinstall -q
echo       완료!

echo.
echo [2/4] MediaPipe 재설치...
pip uninstall mediapipe -y 2>nul
pip install mediapipe --force-reinstall -q
echo       완료!

echo.
echo [3/4] NumPy 호환성 확인...
pip install "numpy<2.0" --force-reinstall -q
echo       완료!

echo.
echo [4/4] protobuf 버전 맞춤...
pip install "protobuf>=3.20,<5.0" --force-reinstall -q
echo       완료!

echo.
echo ============================================================
echo   DLL 문제 해결 완료!
echo   run.bat를 다시 실행해주세요.
echo ============================================================
echo.

:end
pause
