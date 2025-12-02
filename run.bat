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
echo [1/9] Python 확인 중...
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
echo [2/9] Node.js 확인 중...
node --version >nul 2>&1
if errorlevel 1 (
    echo       [오류] Node.js가 설치되어 있지 않습니다!
    echo       https://nodejs.org/ 에서 설치해주세요.
    pause
    exit /b 1
)
echo       Node.js 확인 완료

:: ============================================================
:: 3. Git 확인
:: ============================================================
echo.
echo [3/9] Git 확인 중...
git --version >nul 2>&1
if errorlevel 1 (
    echo       [오류] Git이 설치되어 있지 않습니다!
    echo       https://git-scm.com/downloads 에서 설치해주세요.
    pause
    exit /b 1
)
echo       Git 확인 완료

:: ============================================================
:: 4. NVIDIA GPU 확인 및 PyTorch 설치
:: ============================================================
echo.
echo [4/9] NVIDIA GPU 확인 중...
set HAS_NVIDIA=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo       NVIDIA GPU 감지됨! CUDA 버전으로 설치합니다.
    set HAS_NVIDIA=1
) else (
    echo       NVIDIA GPU 없음. CPU 버전으로 설치합니다.
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
:: 5. 백엔드 패키지 설치
:: ============================================================
echo.
echo [5/9] 백엔드 패키지 확인 중...
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
    pip install mediapipe librosa einops -q

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
:: 6. MuseTalk 설치 (립싱크 모델)
:: ============================================================
echo.
echo [6/9] MuseTalk 립싱크 모델 확인 중...

:: MuseTalk 소스 코드 클론
if not exist "external\MuseTalk\musetalk" (
    echo       MuseTalk 소스 코드 다운로드 중...
    if not exist "external" mkdir external
    cd external

    if exist "MuseTalk" (
        echo       기존 MuseTalk 폴더 삭제 중...
        rmdir /s /q MuseTalk 2>nul
    )

    git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git
    cd ..
    echo       MuseTalk 소스 다운로드 완료!
) else (
    echo       MuseTalk 소스 확인됨
)

:: MuseTalk 의존성 설치
python -c "import mmcv" 2>nul
if errorlevel 1 (
    echo       MuseTalk 의존성 설치 중... (약 5분 소요)

    :: mmcv, mmdet, mmpose 설치 (MuseTalk 필수 의존성)
    pip install openmim -q
    mim install mmengine -q
    mim install "mmcv>=2.0.0" -q
    mim install "mmdet>=3.0.0" -q
    mim install "mmpose>=1.0.0" -q

    :: MuseTalk 추가 의존성
    pip install face-alignment dlib -q
    pip install kornia yacs einops -q

    echo       MuseTalk 의존성 설치 완료!
) else (
    echo       MuseTalk 의존성 확인됨
)

:: MuseTalk 모델 파일 다운로드
if not exist "models\musetalk\musetalkV15\unet.pth" (
    echo       MuseTalk 모델 파일 다운로드 중... (약 1.5GB)

    :: 디렉토리 생성
    if not exist "models\musetalk\musetalkV15" mkdir "models\musetalk\musetalkV15"
    if not exist "models\musetalk\dwpose" mkdir "models\musetalk\dwpose"
    if not exist "models\musetalk\face-parse-bisent" mkdir "models\musetalk\face-parse-bisent"
    if not exist "models\musetalk\whisper" mkdir "models\musetalk\whisper"

    :: HuggingFace에서 모델 다운로드
    python -c "from huggingface_hub import snapshot_download; snapshot_download('TMElyralab/MuseTalk', local_dir='models/musetalk/hf_download', local_dir_use_symlinks=False)"

    :: 파일 복사 (HuggingFace 구조에서 필요한 위치로)
    if exist "models\musetalk\hf_download\models\musetalk\musetalk.json" (
        copy "models\musetalk\hf_download\models\musetalk\musetalk.json" "models\musetalk\musetalkV15\" >nul
    )
    if exist "models\musetalk\hf_download\models\musetalk\pytorch_model.bin" (
        copy "models\musetalk\hf_download\models\musetalk\pytorch_model.bin" "models\musetalk\musetalkV15\unet.pth" >nul
    )

    :: DWPose 모델
    if exist "models\musetalk\hf_download\models\dwpose\dw-ll_ucoco_384.pth" (
        copy "models\musetalk\hf_download\models\dwpose\*" "models\musetalk\dwpose\" >nul
    )

    :: Face parsing 모델
    if exist "models\musetalk\hf_download\models\face-parse-bisent\*" (
        xcopy "models\musetalk\hf_download\models\face-parse-bisent\*" "models\musetalk\face-parse-bisent\" /s /e /y >nul
    )

    :: Whisper 모델
    if exist "models\musetalk\hf_download\models\whisper\*" (
        xcopy "models\musetalk\hf_download\models\whisper\*" "models\musetalk\whisper\" /s /e /y >nul
    )

    echo       MuseTalk 모델 다운로드 완료!
) else (
    echo       MuseTalk 모델 확인됨
)

:: SD-VAE 모델 다운로드
if not exist "models\musetalk\sd-vae-ft-mse\config.json" (
    echo       SD-VAE 모델 다운로드 중...
    python -c "from huggingface_hub import snapshot_download; from pathlib import Path; Path('models/musetalk/sd-vae-ft-mse').mkdir(parents=True,exist_ok=True); snapshot_download('stabilityai/sd-vae-ft-mse',local_dir='models/musetalk/sd-vae-ft-mse')"
    echo       SD-VAE 모델 다운로드 완료!
) else (
    echo       SD-VAE 모델 확인됨
)

:: ============================================================
:: 7. 프론트엔드 패키지 설치
:: ============================================================
echo.
echo [7/9] 프론트엔드 패키지 확인 중...
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
:: 8. 환경설정
:: ============================================================
echo.
echo [8/9] 환경 설정 확인 중...

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
:: 9. 서버 실행 (백엔드 + 프론트엔드)
:: ============================================================
echo.
echo [9/9] 서버 시작 중...
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
