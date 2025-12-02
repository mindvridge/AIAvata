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
echo [1/10] Python 확인 중...
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
echo [2/10] Node.js 확인 중...
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
echo [3/10] Git 확인 중...
git --version >nul 2>&1
if errorlevel 1 (
    echo       [오류] Git이 설치되어 있지 않습니다!
    echo       https://git-scm.com/downloads 에서 설치해주세요.
    pause
    exit /b 1
)
echo       Git 확인 완료

:: ============================================================
:: 3.5. FFmpeg 확인 및 자동 설치 (edge-tts MP3 변환용)
:: ============================================================
echo.
echo [3.5/10] FFmpeg 확인 중...

:: 로컬 설치된 FFmpeg 경로 확인 (이미 다운로드된 경우 스킵)
set "LOCAL_FFMPEG=%~dp0tools\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
if exist "%LOCAL_FFMPEG%" (
    set "PATH=%~dp0tools\ffmpeg\ffmpeg-master-latest-win64-gpl\bin;%PATH%"
    echo       로컬 FFmpeg 발견, 경로 추가됨
    goto ffmpeg_done
)

:: 시스템 FFmpeg 확인
ffmpeg -version >nul 2>&1
if not errorlevel 1 (
    echo       FFmpeg 확인 완료
    goto ffmpeg_done
)

:: FFmpeg가 없으면 설치
echo       FFmpeg가 설치되어 있지 않습니다. 자동 설치를 시도합니다...

:: winget으로 설치 시도 (Windows 10/11)
winget --version >nul 2>&1
if not errorlevel 1 (
    echo       winget으로 FFmpeg 설치 중...
    winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements -h
    if not errorlevel 1 (
        echo       FFmpeg 설치 완료! (재시작 후 적용됩니다)
        goto ffmpeg_done
    ) else (
        echo       winget 설치 실패, 수동 다운로드를 시도합니다...
    )
)

:: winget 실패 시 직접 다운로드
echo       FFmpeg를 직접 다운로드합니다... (약 80MB, 1-2분 소요)

:: tools 폴더에 ffmpeg 다운로드
if not exist "tools" mkdir tools
if not exist "tools\ffmpeg" mkdir tools\ffmpeg

:: curl로 다운로드 시도 (Windows 10 이상 기본 제공, 진행률 표시)
echo       다운로드 중... (진행률이 표시됩니다)
curl -L -o "tools\ffmpeg.zip" "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip" --progress-bar

if exist "tools\ffmpeg.zip" (
    echo       압축 해제 중...
    powershell -Command "Expand-Archive -Path 'tools\ffmpeg.zip' -DestinationPath 'tools\ffmpeg' -Force"
    del "tools\ffmpeg.zip" 2>nul
) else (
    echo       curl 다운로드 실패, PowerShell로 재시도...
    powershell -Command "& { $ProgressPreference='Continue'; Write-Host 'Downloading FFmpeg...'; Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip' -OutFile 'tools\ffmpeg.zip'; Expand-Archive -Path 'tools\ffmpeg.zip' -DestinationPath 'tools\ffmpeg' -Force; Remove-Item 'tools\ffmpeg.zip' -Force }"
)

if exist "tools\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe" (
    echo       FFmpeg 다운로드 완료!

    :: PATH에 추가 (현재 세션)
    set "PATH=%~dp0tools\ffmpeg\ffmpeg-master-latest-win64-gpl\bin;%PATH%"
    echo       현재 세션에 FFmpeg 경로 추가됨
) else (
    echo       [경고] FFmpeg 다운로드 실패
    echo       수동으로 설치해주세요: https://ffmpeg.org/download.html
    echo       또는: choco install ffmpeg
    echo.
    echo       FFmpeg 없이도 실행은 가능하지만 한국어 TTS가 작동하지 않을 수 있습니다.
    timeout /t 5 /nobreak >nul
)

:ffmpeg_done

:: ============================================================
:: 4. NVIDIA GPU 확인 및 PyTorch 설치
:: ============================================================
echo.
echo [4/10] NVIDIA GPU 확인 중...
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
echo [5/10] 백엔드 패키지 확인 중...
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
    pip install edge-tts gtts chatterbox-tts resemble-perth pydub -q
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
echo [6/10] MuseTalk 립싱크 모델 확인 중...

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

    :: HuggingFace에서 모델 다운로드 (전체 repo)
    echo       HuggingFace에서 MuseTalk 모델 다운로드 중...
    python -c "from huggingface_hub import snapshot_download; snapshot_download('TMElyralab/MuseTalk', local_dir='models/musetalk/hf_download', local_dir_use_symlinks=False)"

    :: 다운로드된 구조 확인 및 파일 복사 (다양한 경로 시도)
    :: HuggingFace 구조: models/musetalk/, models/dwpose/, models/face-parse-bisent/, models/whisper/

    :: MuseTalk 메인 모델 (musetalk.json, pytorch_model.bin)
    if exist "models\musetalk\hf_download\models\musetalk\musetalk.json" (
        echo       Copying MuseTalk config from models/musetalk/
        copy "models\musetalk\hf_download\models\musetalk\musetalk.json" "models\musetalk\musetalkV15\" >nul
        copy "models\musetalk\hf_download\models\musetalk\pytorch_model.bin" "models\musetalk\musetalkV15\unet.pth" >nul
    ) else if exist "models\musetalk\hf_download\musetalk\musetalk.json" (
        echo       Copying MuseTalk config from musetalk/
        copy "models\musetalk\hf_download\musetalk\musetalk.json" "models\musetalk\musetalkV15\" >nul
        copy "models\musetalk\hf_download\musetalk\pytorch_model.bin" "models\musetalk\musetalkV15\unet.pth" >nul
    )

    :: DWPose 모델
    if exist "models\musetalk\hf_download\models\dwpose" (
        echo       Copying DWPose models...
        xcopy "models\musetalk\hf_download\models\dwpose\*" "models\musetalk\dwpose\" /s /e /y >nul 2>nul
    ) else if exist "models\musetalk\hf_download\dwpose" (
        xcopy "models\musetalk\hf_download\dwpose\*" "models\musetalk\dwpose\" /s /e /y >nul 2>nul
    )

    :: Face parsing 모델
    if exist "models\musetalk\hf_download\models\face-parse-bisent" (
        echo       Copying Face Parsing models...
        xcopy "models\musetalk\hf_download\models\face-parse-bisent\*" "models\musetalk\face-parse-bisent\" /s /e /y >nul 2>nul
    ) else if exist "models\musetalk\hf_download\face-parse-bisent" (
        xcopy "models\musetalk\hf_download\face-parse-bisent\*" "models\musetalk\face-parse-bisent\" /s /e /y >nul 2>nul
    )

    :: Whisper 모델
    if exist "models\musetalk\hf_download\models\whisper" (
        echo       Copying Whisper models...
        xcopy "models\musetalk\hf_download\models\whisper\*" "models\musetalk\whisper\" /s /e /y >nul 2>nul
    ) else if exist "models\musetalk\hf_download\whisper" (
        xcopy "models\musetalk\hf_download\whisper\*" "models\musetalk\whisper\" /s /e /y >nul 2>nul
    )

    :: 최종 확인
    if exist "models\musetalk\musetalkV15\unet.pth" (
        echo       MuseTalk 모델 다운로드 완료!
    ) else (
        echo       [경고] MuseTalk 모델 파일을 찾을 수 없습니다.
        echo       다운로드된 파일 구조를 확인하세요: models\musetalk\hf_download\
        echo       수동으로 파일을 복사해야 할 수 있습니다.
    )
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
echo [7/10] 프론트엔드 패키지 확인 중...
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
echo [8/10] 환경 설정 확인 중...

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
echo [9/10] 서버 시작 중...
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
