@echo off
chcp 65001 >nul
title AI Avatar Service

echo.
echo ╔═══════════════════════════════════════════════════════════╗
echo ║            AI Avatar Service - Windows                    ║
echo ╚═══════════════════════════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [1/4] 패키지 확인 중...
python -c "import torch,openai,livekit" 2>nul
if errorlevel 1 (
    echo       패키지 설치 중...
    pip install torch torchaudio torchvision -q
    pip install fastapi uvicorn python-dotenv websockets aiofiles pydantic numpy opencv-python-headless -q
    pip install openai anthropic livekit livekit-api -q
    pip install transformers diffusers huggingface_hub -q
    pip install funasr modelscope omegaconf kaldiio -q
    pip install edge-tts gtts chatterbox-tts resemble-perth -q
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
        echo       ※ API 키를 설정해주세요!
    )
) else (
    echo       .env 파일 존재
)

echo.
echo [4/4] 서버 시작...
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   서버 주소: http://localhost:8000
echo   API 문서:  http://localhost:8000/docs
echo   종료: Ctrl+C
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

pause
