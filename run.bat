@echo off
chcp 65001 >nul
title AI Avatar Service

echo.
echo ============================================================
echo            AI Avatar Service - Windows
echo ============================================================
echo.

cd /d "%~dp0"

:: ============================================================
:: 1. Python 확인
:: ============================================================
echo [1/5] Python 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo       [오류] Python이 설치되어 있지 않습니다!
    echo       https://www.python.org/downloads/ 에서 설치해주세요.
    pause
    exit /b 1
)
echo       Python 확인 완료

:: ============================================================
:: 2. 패키지 설치 (처음 한번만)
:: ============================================================
echo.
echo [2/5] 패키지 확인 중...
python -c "import torch,openai,livekit,cv2,mediapipe" 2>nul
if errorlevel 1 (
    echo       패키지 설치 중... (최초 1회)

    :: 기본 패키지
    pip install torch torchaudio torchvision -q
    pip install fastapi uvicorn python-dotenv websockets aiofiles pydantic -q

    :: DLL 충돌 방지를 위한 버전 고정
    pip install "numpy<2.0" -q
    pip install "protobuf>=3.20,<5.0" -q

    :: OpenCV (headless 버전으로 통일)
    pip uninstall opencv-python opencv-contrib-python -y 2>nul
    pip install opencv-python-headless -q

    :: AI/ML 패키지
    pip install openai anthropic livekit livekit-api -q
    pip install transformers diffusers huggingface_hub -q
    pip install funasr modelscope omegaconf kaldiio -q
    pip install edge-tts gtts chatterbox-tts resemble-perth -q
    pip install mediapipe -q

    echo       패키지 설치 완료!
) else (
    echo       패키지 확인 완료
)

:: ============================================================
:: 3. 모델 다운로드 (처음 한번만)
:: ============================================================
echo.
echo [3/5] 모델 확인 중...
if not exist "models\musetalk\sd-vae-ft-mse\config.json" (
    echo       VAE 모델 다운로드 중...
    python -c "from huggingface_hub import snapshot_download; from pathlib import Path; Path('models/musetalk').mkdir(parents=True,exist_ok=True); snapshot_download('stabilityai/sd-vae-ft-mse',local_dir='models/musetalk/sd-vae-ft-mse')"
    echo       모델 다운로드 완료!
) else (
    echo       모델 확인 완료
)

:: ============================================================
:: 4. 환경설정 파일
:: ============================================================
echo.
echo [4/5] 환경 설정 확인 중...
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo       .env 파일 생성됨
        echo.
        echo       ============================================
        echo       [!] .env 파일에 API 키를 설정해주세요!
        echo           - OPENAI_API_KEY 또는
        echo           - ANTHROPIC_API_KEY
        echo       ============================================
        echo.
    )
) else (
    echo       환경 설정 확인 완료
)

:: ============================================================
:: 5. 서버 실행
:: ============================================================
echo.
echo [5/5] 서버 시작 중...
echo.
echo ============================================================
echo   서버 주소: http://localhost:8000
echo   API 문서:  http://localhost:8000/docs
echo   종료: Ctrl+C
echo ============================================================
echo.

python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

pause
