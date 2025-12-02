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
echo [1/6] Python 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo       [오류] Python이 설치되어 있지 않습니다!
    echo       https://www.python.org/downloads/ 에서 설치해주세요.
    pause
    exit /b 1
)
echo       Python 확인 완료

:: ============================================================
:: 2. Node.js 확인
:: ============================================================
echo.
echo [2/6] Node.js 확인 중...
node --version >nul 2>&1
if errorlevel 1 (
    echo       [오류] Node.js가 설치되어 있지 않습니다!
    echo       https://nodejs.org/ 에서 설치해주세요.
    pause
    exit /b 1
)
echo       Node.js 확인 완료

:: ============================================================
:: 3. NVIDIA GPU 확인 및 PyTorch 설치
:: ============================================================
echo.
echo [3/7] NVIDIA GPU 확인 중...
set HAS_NVIDIA=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo       NVIDIA GPU 감지됨! CUDA 버전 PyTorch 설치합니다.
    set HAS_NVIDIA=1
) else (
    echo       NVIDIA GPU 없음. CPU 버전 PyTorch 설치합니다.
)

:: PyTorch CUDA 버전 확인 (이미 CUDA 버전이면 재설치 안함)
set NEED_PYTORCH_REINSTALL=0
if %HAS_NVIDIA%==1 (
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
    if errorlevel 1 (
        set NEED_PYTORCH_REINSTALL=1
        echo       CPU 버전 PyTorch 감지. CUDA 버전으로 업그레이드합니다...
    ) else (
        echo       CUDA PyTorch 이미 설치됨
    )
)

:: ============================================================
:: 4. 백엔드 패키지 설치
:: ============================================================
echo.
echo [4/7] 백엔드 패키지 확인 중...
python -c "import torch,openai,livekit,cv2,mediapipe" 2>nul
if errorlevel 1 (
    echo       패키지 설치 중... (최초 1회, 약 5-10분 소요)

    :: PyTorch 설치 (GPU/CPU 자동 선택)
    if %HAS_NVIDIA%==1 (
        echo       CUDA PyTorch 설치 중... (약 2GB 다운로드)
        pip uninstall torch torchaudio torchvision -y 2>nul
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    ) else (
        pip install torch torchaudio torchvision -q
    )

    pip install fastapi uvicorn python-dotenv websockets aiofiles pydantic -q
    pip install "numpy<2.0" -q
    pip install "protobuf>=3.20,<5.0" -q

    pip uninstall opencv-python opencv-contrib-python -y 2>nul
    pip install opencv-python-headless -q

    pip install openai anthropic livekit livekit-api -q
    pip install transformers diffusers huggingface_hub -q
    pip install funasr modelscope omegaconf kaldiio -q
    pip install edge-tts gtts chatterbox-tts resemble-perth -q
    pip install mediapipe -q

    echo       패키지 설치 완료!
) else (
    echo       패키지 확인 완료
    :: 기존 패키지 있어도 PyTorch CUDA 업그레이드 필요시 실행
    if %NEED_PYTORCH_REINSTALL%==1 (
        echo       CUDA PyTorch로 업그레이드 중... (약 2GB 다운로드)
        pip uninstall torch torchaudio torchvision -y 2>nul
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 -q
        echo       CUDA PyTorch 업그레이드 완료!
    )
)

:: ============================================================
:: 5. 프론트엔드 패키지 설치
:: ============================================================
echo.
echo [5/7] 프론트엔드 패키지 확인 중...
if not exist "frontend\node_modules" (
    echo       npm install 실행 중...
    cd frontend
    npm install --silent
    cd ..
    echo       프론트엔드 패키지 설치 완료!
) else (
    echo       프론트엔드 패키지 확인 완료
)

:: ============================================================
:: 6. 모델 및 환경설정
:: ============================================================
echo.
echo [6/7] 모델 및 환경 설정 확인 중...
if not exist "models\musetalk\sd-vae-ft-mse\config.json" (
    echo       VAE 모델 다운로드 중...
    python -c "from huggingface_hub import snapshot_download; from pathlib import Path; Path('models/musetalk').mkdir(parents=True,exist_ok=True); snapshot_download('stabilityai/sd-vae-ft-mse',local_dir='models/musetalk/sd-vae-ft-mse')"
    echo       모델 다운로드 완료!
) else (
    echo       모델 확인 완료
)

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo       .env 파일 생성됨
    )
)

:: GPU 감지 시 .env의 DEVICE를 cuda로 변경
if %HAS_NVIDIA%==1 (
    findstr /C:"DEVICE=cpu" .env >nul 2>&1
    if not errorlevel 1 (
        powershell -Command "(Get-Content .env) -replace 'DEVICE=cpu', 'DEVICE=cuda' | Set-Content .env"
        echo       .env DEVICE=cuda로 설정됨
    )
)

if not exist "frontend\.env" (
    if exist "frontend\.env.example" (
        copy frontend\.env.example frontend\.env >nul
    )
)
echo       환경 설정 확인 완료

:: ============================================================
:: 7. 서버 실행 (백엔드 + 프론트엔드)
:: ============================================================
echo.
echo [7/7] 서버 시작 중...
echo.
echo ============================================================
echo   백엔드 API:  http://localhost:8000
echo   프론트엔드:   http://localhost:5173
echo   API 문서:    http://localhost:8000/docs
echo   종료: 이 창을 닫으세요
echo ============================================================
echo.

:: 프론트엔드를 새 창에서 실행 (오류 시에도 창 유지)
start "Frontend - AI Avatar" cmd /k "cd /d %~dp0frontend && npm run dev"

:: 잠시 대기 후 브라우저 열기
timeout /t 5 /nobreak >nul
start http://localhost:5173

:: 백엔드 실행 (현재 창)
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

pause
