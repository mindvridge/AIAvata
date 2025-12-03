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

:: eSpeak-ng 항상 확인 및 설치 (Zonos TTS 필수 의존성)
call :check_espeak
if "%ESPEAK_OK%"=="1" (
    echo       eSpeak-ng 확인됨
) else (
    echo       [경고] eSpeak-ng 설치 실패. Zonos TTS가 작동하지 않을 수 있습니다.
)

:: Zonos TTS 설치 확인
python -c "import zonos" 2>nul
if errorlevel 1 (
    call :install_zonos
)
goto after_zonos

:install_zonos
echo       Zonos TTS 설치 중...

:: eSpeak-ng 재확인
if "%ESPEAK_OK%"=="0" (
    echo       [경고] eSpeak-ng 없이 Zonos 설치를 건너뜁니다.
    exit /b
)

:: Zonos GitHub에서 클론 및 설치
if not exist "external\Zonos" (
    echo       Zonos 소스 다운로드 중...
    cd external
    git clone --depth 1 https://github.com/Zyphra/Zonos.git
    cd ..
)

if exist "external\Zonos\setup.py" (
    echo       Zonos 설치 중... (약 2-3분 소요)
    pip install pydub -q
    pip install -e external\Zonos -q
    echo       Zonos TTS 설치 완료!
) else if exist "external\Zonos\pyproject.toml" (
    echo       Zonos 설치 중... (약 2-3분 소요)
    pip install pydub -q
    pip install -e external\Zonos -q
    echo       Zonos TTS 설치 완료!
) else (
    echo       [경고] Zonos 설치 실패. TTS가 제한될 수 있습니다.
)
exit /b

:check_espeak
set "ESPEAK_OK=0"

:: 방법 1: PATH에서 espeak-ng 확인
where espeak-ng >nul 2>&1
if not errorlevel 1 (
    set "ESPEAK_OK=1"
    exit /b
)

:: 방법 2: 기본 설치 경로 확인
if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" (
    set "ESPEAK_OK=1"
    set "PATH=C:\Program Files\eSpeak NG;%PATH%"
    exit /b
)
if exist "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" (
    set "ESPEAK_OK=1"
    set "PATH=C:\Program Files (x86)\eSpeak NG;%PATH%"
    exit /b
)

echo       [참고] eSpeak-ng가 필요합니다. 자동 설치를 시도합니다...

:: 방법 3: winget으로 설치 시도
winget --version >nul 2>&1
if not errorlevel 1 (
    echo       winget으로 eSpeak-ng 설치 시도...
    winget install eSpeak-NG.eSpeak-NG --accept-source-agreements --accept-package-agreements -h >nul 2>&1
    if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" (
        set "ESPEAK_OK=1"
        set "PATH=C:\Program Files\eSpeak NG;%PATH%"
        echo       eSpeak-ng winget 설치 완료!
        exit /b
    )
)

:: 방법 4: GitHub에서 직접 다운로드 및 설치
echo       GitHub에서 eSpeak-ng 다운로드 중...
if not exist "tools" mkdir tools

curl -L -o "tools\espeak-ng.msi" "https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi" --progress-bar

if not exist "tools\espeak-ng.msi" (
    echo       [경고] eSpeak-ng 다운로드 실패.
    exit /b
)

echo       eSpeak-ng 설치 중... (관리자 권한 필요할 수 있음)
msiexec /i "tools\espeak-ng.msi" /quiet /norestart 2>nul
timeout /t 3 /nobreak >nul

if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" (
    set "ESPEAK_OK=1"
    set "PATH=C:\Program Files\eSpeak NG;%PATH%"
    echo       eSpeak-ng 설치 완료!
    del "tools\espeak-ng.msi" 2>nul
    exit /b
)
if exist "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" (
    set "ESPEAK_OK=1"
    set "PATH=C:\Program Files (x86)\eSpeak NG;%PATH%"
    echo       eSpeak-ng 설치 완료!
    del "tools\espeak-ng.msi" 2>nul
    exit /b
)

:: 방법 5: 수동 설치 안내
echo       [참고] 자동 설치 실패. 수동 설치를 시도합니다...
echo       tools\espeak-ng.msi 파일을 더블클릭하여 설치해주세요.
start "" "tools\espeak-ng.msi"
echo       설치 완료 후 Enter를 눌러주세요...
pause >nul

if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" (
    set "ESPEAK_OK=1"
    set "PATH=C:\Program Files\eSpeak NG;%PATH%"
)
if exist "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" (
    set "ESPEAK_OK=1"
    set "PATH=C:\Program Files (x86)\eSpeak NG;%PATH%"
)
exit /b

:after_zonos

:: eSpeak-ng PATH 재확인 (서브루틴에서 설정한 값이 유지되지 않을 수 있음)
if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" (
    set "PATH=C:\Program Files\eSpeak NG;%PATH%"
    set "PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll"
)
if exist "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" (
    set "PATH=C:\Program Files (x86)\eSpeak NG;%PATH%"
    set "PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files (x86)\eSpeak NG\libespeak-ng.dll"
)

python -c "import torch,openai,livekit,cv2,mediapipe" 2>nul
if errorlevel 1 (
    echo       패키지 설치 중... (최초 1회, 약 5-10분 소요)

    :: NumPy 1.x 버전 먼저 설치 (다른 패키지가 NumPy 2.x 설치하지 않도록)
    pip install numpy==1.26.4 --no-cache-dir -q

    :: PyTorch 설치 (GPU/CPU 자동 선택)
    if %HAS_NVIDIA%==1 (
        echo       CUDA PyTorch 설치 중... (약 2GB 다운로드)
        pip uninstall torch torchaudio torchvision -y 2>nul
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 -q
    ) else (
        pip install torch torchaudio torchvision -q
    )

    pip install fastapi uvicorn python-dotenv websockets aiofiles pydantic -q
    pip install "protobuf>=3.20,<5.0" -q

    pip uninstall opencv-python opencv-contrib-python -y 2>nul
    pip install opencv-python-headless -q

    pip install openai livekit livekit-api -q
    pip install transformers diffusers huggingface_hub -q
    pip install funasr modelscope omegaconf kaldiio -q
    pip install zonos pydub -q
    pip install mediapipe librosa einops --no-cache-dir -q

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
:: 5.5 NumPy 1차 호환성 적용 (mediapipe/matplotlib 문제)
:: ============================================================
echo.
echo [5.5/10] NumPy 호환성 확인 중...
python -c "import numpy; v=numpy.__version__; exit(0 if int(v.split('.')[0]) < 2 else 1)" 2>nul
if errorlevel 1 (
    echo       NumPy 2.x 감지됨, 1.x로 다운그레이드 중...
    pip uninstall numpy -y >nul 2>&1
    pip install numpy==1.26.4 --no-cache-dir -q
    echo       NumPy 다운그레이드 완료!
) else (
    echo       NumPy 호환성 확인됨
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

    :: NumPy 1.x 버전 고정 (mmcv가 NumPy 2.x 설치 방지)
    pip install numpy==1.26.4 --no-cache-dir -q

    :: mmcv, mmdet, mmpose 설치 (MuseTalk 필수 의존성)
    pip install openmim -q
    mim install mmengine -q
    mim install "mmcv>=2.0.0" -q
    mim install "mmdet>=3.0.0" -q
    mim install "mmpose>=1.0.0" -q

    :: MuseTalk 추가 의존성
    pip install face-alignment dlib --no-cache-dir -q
    pip install kornia yacs einops --no-cache-dir -q

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

:: Face-parse-bisent 모델 경로 설정 (MuseTalk이 ./models/face-parse-bisent 경로 기대)
:: 두 파일 모두 필요: resnet18-5c106cde.pth, 79999_iter.pth
call :download_face_parser
goto after_face_parser

:download_face_parser
:: 디렉토리 생성
if not exist "models\face-parse-bisent" mkdir "models\face-parse-bisent"

:: 파일 검증 (손상된 파일 삭제)
if exist "models\face-parse-bisent\79999_iter.pth" (
    python -c "import torch; torch.load('models/face-parse-bisent/79999_iter.pth', map_location='cpu', weights_only=True)" 2>nul
    if errorlevel 1 (
        echo       [경고] 79999_iter.pth 손상 감지, 삭제 후 재다운로드...
        del "models\face-parse-bisent\79999_iter.pth" 2>nul
    ) else (
        echo       Face parser 모델 확인됨 (검증 완료)
        exit /b
    )
)

:: resnet18 다운로드 (없으면)
if not exist "models\face-parse-bisent\resnet18-5c106cde.pth" (
    echo       resnet18 모델 다운로드 중...
    curl -L -o "models\face-parse-bisent\resnet18-5c106cde.pth" "https://download.pytorch.org/models/resnet18-5c106cde.pth" --progress-bar
)

:: 79999_iter.pth 다운로드 (없으면)
if not exist "models\face-parse-bisent\79999_iter.pth" (
    echo       79999_iter.pth 모델 다운로드 중...

    :: 방법 1: MuseTalk HF repo에서 직접 다운로드 (huggingface_hub 사용)
    echo       방법 1: HuggingFace Hub에서 다운로드...
    python -c "from huggingface_hub import hf_hub_download; import shutil; import os; f=hf_hub_download(repo_id='TMElyralab/MuseTalk', filename='models/face-parse-bisent/79999_iter.pth'); os.makedirs('models/face-parse-bisent', exist_ok=True); shutil.copy(f, 'models/face-parse-bisent/79999_iter.pth'); print('Downloaded:', f)" 2>nul
)

:: 방법 1 실패 시 - 방법 2: 직접 URL 다운로드
if not exist "models\face-parse-bisent\79999_iter.pth" (
    echo       방법 2: HuggingFace 직접 URL에서 다운로드...
    curl -L -o "models\face-parse-bisent\79999_iter.pth" "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/models/face-parse-bisent/79999_iter.pth" --progress-bar
)

:: 방법 2 실패 시 - 방법 3: 이미 다운로드된 musetalk 폴더에서 복사
if not exist "models\face-parse-bisent\79999_iter.pth" (
    if exist "models\musetalk\face-parse-bisent\79999_iter.pth" (
        echo       방법 3: 기존 MuseTalk 폴더에서 복사...
        copy "models\musetalk\face-parse-bisent\79999_iter.pth" "models\face-parse-bisent\79999_iter.pth" >nul
    )
)

:: 방법 3 실패 시 - 방법 4: HF 다운로드 폴더에서 복사
if not exist "models\face-parse-bisent\79999_iter.pth" (
    if exist "models\musetalk\hf_download\models\face-parse-bisent\79999_iter.pth" (
        echo       방법 4: HF 다운로드 폴더에서 복사...
        copy "models\musetalk\hf_download\models\face-parse-bisent\79999_iter.pth" "models\face-parse-bisent\79999_iter.pth" >nul
    )
)

:: 다운로드 성공 검증
if exist "models\face-parse-bisent\79999_iter.pth" (
    python -c "import torch; torch.load('models/face-parse-bisent/79999_iter.pth', map_location='cpu', weights_only=True)" 2>nul
    if errorlevel 1 (
        echo       [경고] 다운로드된 파일 손상, 삭제 중...
        del "models\face-parse-bisent\79999_iter.pth" 2>nul
    ) else (
        echo       Face parser 모델 설정 완료!
        exit /b
    )
)

if not exist "models\face-parse-bisent\79999_iter.pth" (
    echo       [경고] Face parser 모델 다운로드 실패. 립싱크 품질이 저하될 수 있습니다.
    echo       수동 다운로드: https://huggingface.co/TMElyralab/MuseTalk/tree/main/models/face-parse-bisent
)
exit /b

:after_face_parser

:: ============================================================
:: 6.5 LivePortrait 설치 (Idle 애니메이션)
:: ============================================================
echo.
echo [6.5/10] LivePortrait 설치 확인 중...

:: LivePortrait 소스 코드 클론
if not exist "external\LivePortrait\src" (
    echo       LivePortrait 소스 코드 다운로드 중...
    if not exist "external" mkdir external
    cd external

    if exist "LivePortrait" (
        rmdir /s /q LivePortrait 2>nul
    )

    git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git
    cd ..
    echo       LivePortrait 소스 다운로드 완료!
) else (
    echo       LivePortrait 소스 확인됨
)

:: LivePortrait 의존성 설치
python -c "import onnxruntime" 2>nul
if errorlevel 1 (
    echo       LivePortrait 의존성 설치 중...
    :: NumPy 1.x 버전 고정 (onnxruntime가 NumPy 2.x 설치 방지)
    pip install numpy==1.26.4 --no-cache-dir -q
    pip install onnxruntime-gpu onnx --no-cache-dir -q
    pip install tyro rich tqdm --no-cache-dir -q
    echo       LivePortrait 의존성 설치 완료!
)

:: LivePortrait 모델 파일 다운로드
if not exist "models\live_portrait\appearance_feature_extractor.safetensors" (
    echo       LivePortrait 모델 파일 다운로드 중... (약 400MB)

    if not exist "models\live_portrait" mkdir "models\live_portrait"

    :: HuggingFace에서 LivePortrait 모델 다운로드
    python -c "from huggingface_hub import snapshot_download; snapshot_download('KwaiVGI/LivePortrait', local_dir='models/live_portrait', local_dir_use_symlinks=False)"
    echo       LivePortrait 모델 다운로드 완료!
) else (
    echo       LivePortrait 모델 확인됨
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
:: 8.5 NumPy 최종 호환성 강제 적용 (모든 패키지 설치 후)
:: ============================================================
echo.
echo [8.5/10] NumPy 최종 호환성 강제 적용 중...
python -c "import numpy; v=numpy.__version__; exit(0 if int(v.split('.')[0]) < 2 else 1)" 2>nul
if errorlevel 1 (
    echo       [!] NumPy 2.x가 다시 설치됨, 최종 강제 다운그레이드 중...
    pip uninstall numpy -y >nul 2>&1
    pip cache purge >nul 2>&1
    pip install numpy==1.26.4 --no-cache-dir --force-reinstall -q

    :: 의존성 패키지도 재컴파일 (NumPy 헤더 호환성)
    echo       mediapipe/matplotlib 재설치 중...
    pip uninstall mediapipe -y >nul 2>&1
    pip install mediapipe --no-cache-dir -q

    python -c "import numpy; print(f'       NumPy 버전: {numpy.__version__}')"
    echo       NumPy 호환성 강제 적용 완료!
) else (
    python -c "import numpy; print(f'       NumPy 버전: {numpy.__version__} [호환]')"
)

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

:: eSpeak-ng PATH 설정 (Zonos TTS용)
set "ESPEAK_PATH="
if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" set "ESPEAK_PATH=C:\Program Files\eSpeak NG"
if exist "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" set "ESPEAK_PATH=C:\Program Files (x86)\eSpeak NG"

:: 백엔드를 새 창에서 실행 (eSpeak-ng PATH 포함, torch.compile 비활성화)
:: TORCHDYNAMO_DISABLE=1: Windows에서 Triton 경고 방지
if defined ESPEAK_PATH (
    start "Backend - AI Avatar" cmd /k "cd /d %~dp0 && set PATH=%ESPEAK_PATH%;%PATH% && set PHONEMIZER_ESPEAK_LIBRARY=%ESPEAK_PATH%\libespeak-ng.dll && set TORCHDYNAMO_DISABLE=1 && python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"
) else (
    echo       [경고] eSpeak-ng가 설치되지 않았습니다. Zonos TTS가 작동하지 않을 수 있습니다.
    start "Backend - AI Avatar" cmd /k "cd /d %~dp0 && set TORCHDYNAMO_DISABLE=1 && python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload"
)

:: 백엔드 서버가 준비될 때까지 대기
echo       백엔드 서버 초기화 대기 중...
:wait_backend
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo       ... 백엔드 초기화 중 ...
    goto wait_backend
)
echo       백엔드 서버 준비 완료!

:: 프론트엔드를 새 창에서 실행
start "Frontend - AI Avatar" cmd /k "cd /d %~dp0frontend && npm run dev"

:: 프론트엔드 서버가 준비될 때까지 대기
echo       프론트엔드 서버 초기화 대기 중...
:wait_frontend
timeout /t 2 /nobreak >nul
curl -s http://localhost:5173 >nul 2>&1
if errorlevel 1 (
    echo       ... 프론트엔드 초기화 중 ...
    goto wait_frontend
)
echo       프론트엔드 서버 준비 완료!

:: 모든 서버 준비 후 브라우저 열기
echo.
echo [10/10] 브라우저 열기...
start http://localhost:5173

echo.
echo ============================================================
echo   모든 서비스가 시작되었습니다!
echo   브라우저가 자동으로 열렸습니다.
echo   종료하려면 이 창과 서버 창들을 닫으세요.
echo ============================================================
echo.
pause
