@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title AI Avatar Service

echo.
echo ============================================================
echo            AI Avatar Service - Windows
echo ============================================================
echo.
echo [TIP] If you see permission errors, close other Python programs first.
echo.

cd /d "%~dp0"

REM ============================================================
REM 0. Cleanup corrupted packages and set pip timeout
REM ============================================================
echo [0/10] 환경 정리 중...

REM Set pip timeout to prevent network errors (default 15 -> 120 seconds)
set PIP_DEFAULT_TIMEOUT=120

REM Get site-packages path (check multiple Python installations)
for /f "tokens=*" %%i in ('python -c "import site; print(site.getsitepackages()[0])" 2^>nul') do set SITE_PACKAGES=%%i

REM Also check Python 3.10 path (common installation location)
set PYTHON310_PATH=C:\Users\pc\AppData\Local\Programs\Python\Python310\Lib\site-packages

REM Clean up ALL corrupted/temp folders (more aggressive cleanup)
if defined SITE_PACKAGES (
    echo       손상된 패키지 정리 중...

    REM Remove corrupted -umpy folder
    if exist "%SITE_PACKAGES%\-umpy" (
        echo       삭제: -umpy
        rmdir /s /q "%SITE_PACKAGES%\-umpy" 2>nul
    )

    REM Remove any temp folders starting with ~ or -
    for /d %%d in ("%SITE_PACKAGES%\~*") do (
        echo       삭제: %%~nxd
        rmdir /s /q "%%d" 2>nul
    )
    for /d %%d in ("%SITE_PACKAGES%\-*") do (
        echo       삭제: %%~nxd
        rmdir /s /q "%%d" 2>nul
    )

    REM Remove .dist-info for corrupted packages
    for /d %%d in ("%SITE_PACKAGES%\~*.dist-info") do rmdir /s /q "%%d" 2>nul
)

REM Also clean Python 3.10 path if it exists
if exist "%PYTHON310_PATH%\-umpy" (
    echo       Python 3.10 경로 정리 중...
    rmdir /s /q "%PYTHON310_PATH%\-umpy" 2>nul
    for /d %%d in ("%PYTHON310_PATH%\~*") do rmdir /s /q "%%d" 2>nul
    for /d %%d in ("%PYTHON310_PATH%\-*") do rmdir /s /q "%%d" 2>nul
)

REM Upgrade pip to avoid old version issues
python -m pip install --upgrade pip -q 2>nul

echo       환경 정리 완료

REM ============================================================
REM 0.5 Git pull and Python cache cleanup
REM ============================================================
echo.
echo [0.5/10] 최신 코드 업데이트 및 캐시 정리 중...

REM Pull latest code from git (if git is available)
git --version >nul 2>&1
if not errorlevel 1 (
    echo       Git에서 최신 코드 가져오는 중...
    git pull 2>nul
    if not errorlevel 1 (
        echo       ✅ 최신 코드 업데이트 완료
    ) else (
        echo       ⚠️ Git pull 실패 (로컬 변경사항이 있거나 네트워크 문제)
    )
) else (
    echo       ⚠️ Git이 설치되지 않아 자동 업데이트 건너뜀
)

REM Clean Python cache (to ensure latest code is used)
echo       Python 캐시 정리 중...
if exist "__pycache__" rmdir /s /q "__pycache__" 2>nul
if exist "src\__pycache__" rmdir /s /q "src\__pycache__" 2>nul
if exist "src\api\__pycache__" rmdir /s /q "src\api\__pycache__" 2>nul
if exist "src\pipeline\__pycache__" rmdir /s /q "src\pipeline\__pycache__" 2>nul
if exist "src\models\__pycache__" rmdir /s /q "src\models\__pycache__" 2>nul
if exist "src\models\integrations\__pycache__" rmdir /s /q "src\models\integrations\__pycache__" 2>nul
if exist "src\services\__pycache__" rmdir /s /q "src\services\__pycache__" 2>nul
if exist "src\utils\__pycache__" rmdir /s /q "src\utils\__pycache__" 2>nul
echo       ✅ Python 캐시 정리 완료

REM ============================================================
REM 1. Python check
REM ============================================================
echo [1/10] Python 확인 중...
python --version >nul 2>&1
if errorlevel 1 (
    echo       [오류] Python이 설치되어 있지 않습니다!
    echo       https://www.python.org/downloads/ 에서 설치해주세요.
    pause
    exit /b 1
)
echo       Python 확인 완료

REM ============================================================
REM 2. Node.js check
REM ============================================================
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

REM ============================================================
REM 3. Git check
REM ============================================================
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

REM ============================================================
REM 3.5. FFmpeg check and auto-install (for edge-tts MP3 conversion)
REM ============================================================
echo.
echo [3.5/10] FFmpeg 확인 중...

REM Check local FFmpeg path (skip if already downloaded)
set "LOCAL_FFMPEG=%~dp0tools\ffmpeg\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe"
if exist "%LOCAL_FFMPEG%" (
    set "PATH=%~dp0tools\ffmpeg\ffmpeg-master-latest-win64-gpl\bin;%PATH%"
    echo       로컬 FFmpeg 발견, 경로 추가됨
    goto ffmpeg_done
)

REM Check system FFmpeg
ffmpeg -version >nul 2>&1
if not errorlevel 1 (
    echo       FFmpeg 확인 완료
    goto ffmpeg_done
)

REM Install FFmpeg if not found
echo       FFmpeg가 설치되어 있지 않습니다. 자동 설치를 시도합니다...

REM Try installing with winget (Windows 10/11)
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

REM Direct download if winget fails
echo       FFmpeg를 직접 다운로드합니다... (약 80MB, 1-2분 소요)

REM Download ffmpeg to tools folder
if not exist "tools" mkdir tools
if not exist "tools\ffmpeg" mkdir tools\ffmpeg

REM Try download with curl (built-in on Windows 10+, shows progress)
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

    REM Add to PATH (current session)
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

REM ============================================================
REM 4. NVIDIA GPU check and PyTorch installation
REM ============================================================
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

REM Check PyTorch CUDA version (skip reinstall if already CUDA version)
set NEED_PYTORCH_REINSTALL=0
set PYTORCH_INSTALLED=0
if %HAS_NVIDIA%==1 (
    REM First check if torch is installed at all
    python -c "import torch; print(torch.__version__)" >nul 2>&1
    if not errorlevel 1 (
        set PYTORCH_INSTALLED=1
        REM Then check if CUDA is available
        python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
        if errorlevel 1 (
            set NEED_PYTORCH_REINSTALL=1
            echo       CPU 버전 PyTorch 감지. CUDA 버전으로 업그레이드합니다...
        ) else (
            echo       CUDA PyTorch 이미 설치됨
        )
    ) else (
        echo       PyTorch 미설치. CUDA 버전으로 설치합니다...
    )
)

REM ============================================================
REM 5. Backend package installation
REM ============================================================
echo.
echo [5/10] 백엔드 패키지 확인 중...

REM Always check and install eSpeak-ng (required for Zonos TTS)
call :check_espeak
if "%ESPEAK_OK%"=="1" (
    echo       eSpeak-ng 확인됨
) else (
    echo       [경고] eSpeak-ng 설치 실패. Zonos TTS가 작동하지 않을 수 있습니다.
)

REM Check Zonos TTS installation
python -c "import zonos" 2>nul
if errorlevel 1 (
    call :install_zonos
)
goto after_zonos

:install_zonos
echo       Zonos TTS 설치 중...

REM Re-check eSpeak-ng
if "%ESPEAK_OK%"=="0" (
    echo       [경고] eSpeak-ng 없이 Zonos 설치를 건너뜁니다.
    exit /b
)

REM Clone and install from Zonos GitHub
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

REM Method 1: Check espeak-ng in PATH
where espeak-ng >nul 2>&1
if not errorlevel 1 (
    set "ESPEAK_OK=1"
    exit /b
)

REM Method 2: Check default installation path
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

REM Method 3: Try installing with winget
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

REM Method 4: Direct download and install from GitHub
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

REM Method 5: Manual installation guide
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

REM Re-check eSpeak-ng PATH (value from subroutine may not persist)
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

    REM Install NumPy 1.x first (prevent other packages from installing NumPy 2.x)
    pip install numpy==1.26.4 --no-cache-dir -q 2>nul || pip install numpy==1.26.4 --no-cache-dir --user -q

    REM Install PyTorch (auto-select GPU/CPU)
    if %HAS_NVIDIA%==1 (
        if %PYTORCH_INSTALLED%==0 (
            echo       CUDA PyTorch 설치 중... (약 2GB 다운로드)
            pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 -q
        )
    ) else (
        pip install torch torchaudio torchvision -q
    )

    pip install fastapi uvicorn python-dotenv websockets aiofiles pydantic -q
    pip install "protobuf>=4.25.3,<5.0.0" -q

    REM Install opencv-python compatible with NumPy 1.x (4.8.x supports numpy<2)
    pip uninstall opencv-python-headless opencv-python opencv-contrib-python -y 2>nul
    pip install opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78 -q 2>nul || pip install opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78 --user -q

    pip install openai livekit livekit-api -q
    pip install transformers diffusers huggingface_hub -q
    pip install funasr modelscope omegaconf kaldiio -q

    REM Install zonos with --no-deps to avoid NumPy 2.x, then install missing deps manually
    pip install pydub -q
    pip install zonos --no-deps -q 2>nul
    pip install scipy einops -q

    pip install mediapipe librosa --no-cache-dir -q

    REM Final NumPy pin (ensure 1.x after all installs)
    pip install numpy==1.26.4 --no-cache-dir -q 2>nul

    echo       패키지 설치 완료!
) else (
    echo       패키지 확인 완료
    REM Run PyTorch CUDA upgrade only if needed (not in a loop)
    if %NEED_PYTORCH_REINSTALL%==1 (
        if %PYTORCH_INSTALLED%==1 (
            echo       CUDA PyTorch로 업그레이드 중... (약 2GB 다운로드)
            pip uninstall torch torchaudio torchvision -y 2>nul
            pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121 -q
            echo       CUDA PyTorch 업그레이드 완료!
        )
    )
)

REM ============================================================
REM 5.5 NumPy first compatibility fix (mediapipe/matplotlib issue)
REM ============================================================
echo.
echo [5.5/10] NumPy 호환성 확인 중...
python -c "import numpy; v=numpy.__version__; exit(0 if int(v.split('.')[0]) < 2 else 1)" 2>nul
if errorlevel 1 (
    echo       NumPy 2.x 감지됨, 1.x로 다운그레이드 중...
    pip uninstall numpy -y >nul 2>&1
    pip install numpy==1.26.4 --no-cache-dir -q 2>nul
    if errorlevel 1 (
        echo       [경고] 권한 오류 발생, --user 옵션으로 재시도 중...
        pip install numpy==1.26.4 --no-cache-dir --user -q
    )
    echo       NumPy 다운그레이드 완료!
) else (
    echo       NumPy 호환성 확인됨
)

REM ============================================================
REM 6. MuseTalk installation (lip-sync model)
REM ============================================================
echo.
echo [6/10] MuseTalk 립싱크 모델 확인 중...

REM Clone MuseTalk source code
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

REM Install MuseTalk dependencies
python -c "import mmcv" 2>nul
if errorlevel 1 (
    echo       MuseTalk 의존성 설치 중... (약 5분 소요)

    REM Pin NumPy 1.x (prevent mmcv from installing NumPy 2.x)
    pip install numpy==1.26.4 --no-cache-dir -q

    REM Install mmcv, mmdet, mmpose (MuseTalk required dependencies)
    pip install openmim -q
    python -m mim install mmengine -q
    python -m mim install "mmcv>=2.0.0" -q
    python -m mim install "mmdet>=3.0.0" -q
    python -m mim install "mmpose>=1.0.0" -q

    REM Check and install CMake for dlib build
    cmake --version >nul 2>&1
    if errorlevel 1 (
        echo       CMake 설치 중... (dlib 빌드에 필요)
        winget --version >nul 2>&1
        if not errorlevel 1 (
            winget install Kitware.CMake --accept-source-agreements --accept-package-agreements -h >nul 2>&1
        )
        REM If winget fails, try pip cmake
        cmake --version >nul 2>&1
        if errorlevel 1 (
            pip install cmake -q
        )
    )

    REM MuseTalk additional dependencies
    pip install face-alignment --no-cache-dir -q
    REM Try pre-built dlib wheel first (faster, no CMake needed)
    pip install dlib --no-cache-dir -q 2>nul
    if errorlevel 1 (
        echo       [경고] dlib 사전 빌드 휠 없음, 소스에서 빌드 중...
        pip install dlib --no-cache-dir -q
    )
    pip install kornia yacs einops --no-cache-dir -q

    echo       MuseTalk 의존성 설치 완료!
) else (
    echo       MuseTalk 의존성 확인됨
)

REM Download MuseTalk model files
if not exist "models\musetalk\musetalkV15\unet.pth" (
    echo       MuseTalk 모델 파일 다운로드 중... (약 1.5GB)

    REM Create directories
    if not exist "models\musetalk\musetalkV15" mkdir "models\musetalk\musetalkV15"
    if not exist "models\musetalk\dwpose" mkdir "models\musetalk\dwpose"
    if not exist "models\musetalk\face-parse-bisent" mkdir "models\musetalk\face-parse-bisent"
    if not exist "models\musetalk\whisper" mkdir "models\musetalk\whisper"

    REM Download model from HuggingFace (full repo)
    echo       HuggingFace에서 MuseTalk 모델 다운로드 중...
    python -c "from huggingface_hub import snapshot_download; snapshot_download('TMElyralab/MuseTalk', local_dir='models/musetalk/hf_download', local_dir_use_symlinks=False)"

    REM Check downloaded structure and copy files (try various paths)
    REM HuggingFace structure: models/musetalk/, models/dwpose/, models/face-parse-bisent/, models/whisper/

    REM MuseTalk main model (musetalk.json, pytorch_model.bin)
    if exist "models\musetalk\hf_download\models\musetalk\musetalk.json" (
        echo       Copying MuseTalk config from models/musetalk/
        copy "models\musetalk\hf_download\models\musetalk\musetalk.json" "models\musetalk\musetalkV15\" >nul
        copy "models\musetalk\hf_download\models\musetalk\pytorch_model.bin" "models\musetalk\musetalkV15\unet.pth" >nul
    ) else if exist "models\musetalk\hf_download\musetalk\musetalk.json" (
        echo       Copying MuseTalk config from musetalk/
        copy "models\musetalk\hf_download\musetalk\musetalk.json" "models\musetalk\musetalkV15\" >nul
        copy "models\musetalk\hf_download\musetalk\pytorch_model.bin" "models\musetalk\musetalkV15\unet.pth" >nul
    )

    REM DWPose model
    if exist "models\musetalk\hf_download\models\dwpose" (
        echo       Copying DWPose models...
        xcopy "models\musetalk\hf_download\models\dwpose\*" "models\musetalk\dwpose\" /s /e /y >nul 2>nul
    ) else if exist "models\musetalk\hf_download\dwpose" (
        xcopy "models\musetalk\hf_download\dwpose\*" "models\musetalk\dwpose\" /s /e /y >nul 2>nul
    )

    REM Face parsing model
    if exist "models\musetalk\hf_download\models\face-parse-bisent" (
        echo       Copying Face Parsing models...
        xcopy "models\musetalk\hf_download\models\face-parse-bisent\*" "models\musetalk\face-parse-bisent\" /s /e /y >nul 2>nul
    ) else if exist "models\musetalk\hf_download\face-parse-bisent" (
        xcopy "models\musetalk\hf_download\face-parse-bisent\*" "models\musetalk\face-parse-bisent\" /s /e /y >nul 2>nul
    )

    REM Whisper model
    if exist "models\musetalk\hf_download\models\whisper" (
        echo       Copying Whisper models...
        xcopy "models\musetalk\hf_download\models\whisper\*" "models\musetalk\whisper\" /s /e /y >nul 2>nul
    ) else if exist "models\musetalk\hf_download\whisper" (
        xcopy "models\musetalk\hf_download\whisper\*" "models\musetalk\whisper\" /s /e /y >nul 2>nul
    )

    REM Final verification
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

REM Download SD-VAE model
if not exist "models\musetalk\sd-vae-ft-mse\config.json" (
    echo       SD-VAE 모델 다운로드 중...
    python -c "from huggingface_hub import snapshot_download; from pathlib import Path; Path('models/musetalk/sd-vae-ft-mse').mkdir(parents=True,exist_ok=True); snapshot_download('stabilityai/sd-vae-ft-mse',local_dir='models/musetalk/sd-vae-ft-mse')"
    echo       SD-VAE 모델 다운로드 완료!
) else (
    echo       SD-VAE 모델 확인됨
)

REM Set face-parse-bisent model path (MuseTalk expects ./models/face-parse-bisent)
REM Both files required: resnet18-5c106cde.pth, 79999_iter.pth
call :download_face_parser
goto after_face_parser

:download_face_parser
REM Create directory
if not exist "models\face-parse-bisent" mkdir "models\face-parse-bisent"

REM Verify file (delete if corrupted)
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

REM Download resnet18 (if not present)
if not exist "models\face-parse-bisent\resnet18-5c106cde.pth" (
    echo       resnet18 모델 다운로드 중...
    curl -L -o "models\face-parse-bisent\resnet18-5c106cde.pth" "https://download.pytorch.org/models/resnet18-5c106cde.pth" --progress-bar
)

REM Download 79999_iter.pth (if not present)
if not exist "models\face-parse-bisent\79999_iter.pth" (
    echo       79999_iter.pth 모델 다운로드 중...

    REM Method 1: Download from vivym/face-parsing-bisenet (reliable alternative repo)
    echo       방법 1: HuggingFace vivym/face-parsing-bisenet에서 다운로드...
    python -c "from huggingface_hub import hf_hub_download; import shutil; import os; f=hf_hub_download(repo_id='vivym/face-parsing-bisenet', filename='79999_iter.pth'); os.makedirs('models/face-parse-bisent', exist_ok=True); shutil.copy(f, 'models/face-parse-bisent/79999_iter.pth'); print('Downloaded:', f)" 2>nul
)

REM If method 1 fails - Method 2: Direct URL from vivym repo
if not exist "models\face-parse-bisent\79999_iter.pth" (
    echo       방법 2: HuggingFace 직접 URL에서 다운로드...
    curl -L -o "models\face-parse-bisent\79999_iter.pth" "https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth" --progress-bar
    REM Check file size (should be >50MB, not HTML error page)
    for %%A in ("models\face-parse-bisent\79999_iter.pth") do (
        if %%~zA LSS 1000000 (
            echo       [경고] 다운로드 파일 크기 이상, 삭제 중...
            del "models\face-parse-bisent\79999_iter.pth" 2>nul
        )
    )
)

REM If method 2 fails - Method 2.5: Try ManyOtherFunctions repo (another mirror)
if not exist "models\face-parse-bisent\79999_iter.pth" (
    echo       방법 2.5: ManyOtherFunctions 미러에서 다운로드...
    curl -L -o "models\face-parse-bisent\79999_iter.pth" "https://huggingface.co/ManyOtherFunctions/face-parse-bisent/resolve/main/79999_iter.pth" --progress-bar
    for %%A in ("models\face-parse-bisent\79999_iter.pth") do (
        if %%~zA LSS 1000000 (
            del "models\face-parse-bisent\79999_iter.pth" 2>nul
        )
    )
)

REM If method 2.5 fails - Method 3: Copy from already downloaded musetalk folder
if not exist "models\face-parse-bisent\79999_iter.pth" (
    if exist "models\musetalk\face-parse-bisent\79999_iter.pth" (
        echo       방법 3: 기존 MuseTalk 폴더에서 복사...
        copy "models\musetalk\face-parse-bisent\79999_iter.pth" "models\face-parse-bisent\79999_iter.pth" >nul
    )
)

REM If method 3 fails - Method 4: Copy from HF download folder
if not exist "models\face-parse-bisent\79999_iter.pth" (
    if exist "models\musetalk\hf_download\models\face-parse-bisent\79999_iter.pth" (
        echo       방법 4: HF 다운로드 폴더에서 복사...
        copy "models\musetalk\hf_download\models\face-parse-bisent\79999_iter.pth" "models\face-parse-bisent\79999_iter.pth" >nul
    )
)

REM Verify download success
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
    echo       수동 다운로드: https://huggingface.co/vivym/face-parsing-bisenet/blob/main/79999_iter.pth
    echo       다운로드 후 models\face-parse-bisent\ 폴더에 저장하세요.
)
exit /b

:after_face_parser

REM ============================================================
REM 6.5 LivePortrait installation (Idle animation)
REM ============================================================
echo.
echo [6.5/10] LivePortrait 설치 확인 중...

REM Clone LivePortrait source code
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

    REM Create __init__.py files for package import (required for Python imports)
    echo       LivePortrait 패키지 초기화 파일 생성 중...
    if exist "external\LivePortrait\src" (
        type nul > "external\LivePortrait\src\__init__.py"
        type nul > "external\LivePortrait\src\config\__init__.py"
        type nul > "external\LivePortrait\src\utils\__init__.py"
        type nul > "external\LivePortrait\src\modules\__init__.py"
        echo       __init__.py 파일 생성 완료!
    )
) else (
    echo       LivePortrait 소스 확인됨
    REM Ensure __init__.py files exist
    if not exist "external\LivePortrait\src\__init__.py" (
        type nul > "external\LivePortrait\src\__init__.py"
        type nul > "external\LivePortrait\src\config\__init__.py"
        type nul > "external\LivePortrait\src\utils\__init__.py"
        type nul > "external\LivePortrait\src\modules\__init__.py"
        echo       __init__.py 파일 생성됨
    )
)

REM Install LivePortrait dependencies (simpler approach - check key modules)
echo       LivePortrait 의존성 확인 중...

REM Check pykalman (required)
python -c "import pykalman" 2>nul
if errorlevel 1 goto install_lp_deps

REM Check tyro (required)
python -c "import tyro" 2>nul
if errorlevel 1 goto install_lp_deps

REM Check onnxruntime (required)
python -c "import onnxruntime" 2>nul
if errorlevel 1 goto install_lp_deps

REM All required dependencies OK
echo       LivePortrait 의존성 확인됨

REM Optional: Try to install insightface if missing (use pre-built wheel for Windows)
python -c "import insightface" 2>nul
if errorlevel 1 (
    echo       insightface 설치 중... (사전 빌드된 휠 사용)

    REM Detect Python version (3.9, 3.10, 3.11, 3.12, 3.13)
    for /f "tokens=2 delims=." %%a in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_MINOR=%%a
    set PY_VER=cp3!PY_MINOR!

    echo       Python 버전 감지: 3.!PY_MINOR! (!PY_VER!)

    if not exist "tools" mkdir tools
    set "WHEEL_URL=https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-!PY_VER!-!PY_VER!-win_amd64.whl"
    set "WHEEL_FILE=tools\insightface-!PY_VER!.whl"

    curl -L -o "!WHEEL_FILE!" "!WHEEL_URL!" --progress-bar 2>nul

    REM Check if downloaded file is valid (at least 1MB)
    if exist "!WHEEL_FILE!" (
        for %%A in ("!WHEEL_FILE!") do set WHEEL_SIZE=%%~zA
        if !WHEEL_SIZE! GTR 1000000 (
            pip install "!WHEEL_FILE!" -q
            if not errorlevel 1 (
                echo       insightface 설치 완료!
                del "!WHEEL_FILE!" 2>nul
            ) else (
                echo       [경고] insightface 휠 설치 실패. pip로 재시도...
                pip install insightface --prefer-binary -q 2>nul
            )
        ) else (
            echo       [경고] insightface 휠 다운로드 불완전. pip로 재시도...
            del "!WHEEL_FILE!" 2>nul
            pip install insightface --prefer-binary -q 2>nul
        )
    ) else (
        echo       [경고] insightface 휠 다운로드 실패. pip로 재시도...
        pip install insightface --prefer-binary -q 2>nul
    )
)

goto lp_deps_done

:install_lp_deps
echo       LivePortrait 의존성 설치 중...
REM Pin NumPy 1.x (prevent onnxruntime from installing NumPy 2.x)
pip install numpy==1.26.4 --no-cache-dir -q
pip install onnxruntime-gpu onnx --no-cache-dir -q
pip install tyro rich tqdm --no-cache-dir -q
pip install imageio imageio-ffmpeg --no-cache-dir -q
pip install pykalman --no-cache-dir -q

REM insightface: Download pre-built wheel for Windows (avoid C++ compile errors)
echo       insightface 설치 중... (사전 빌드된 휠 사용)

REM Detect Python version (3.9, 3.10, 3.11, 3.12, 3.13)
for /f "tokens=2 delims=." %%a in ('python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PY_MINOR=%%a
set PY_VER=cp3!PY_MINOR!
echo       Python 버전 감지: 3.!PY_MINOR! (!PY_VER!)

if not exist "tools" mkdir tools
set "WHEEL_URL=https://github.com/Gourieff/Assets/raw/main/Insightface/insightface-0.7.3-!PY_VER!-!PY_VER!-win_amd64.whl"
set "WHEEL_FILE=tools\insightface-!PY_VER!.whl"

curl -L -o "!WHEEL_FILE!" "!WHEEL_URL!" --progress-bar 2>nul

REM Check if downloaded file is valid (at least 1MB)
if exist "!WHEEL_FILE!" (
    for %%A in ("!WHEEL_FILE!") do set WHEEL_SIZE=%%~zA
    if !WHEEL_SIZE! GTR 1000000 (
        pip install "!WHEEL_FILE!" -q
        if not errorlevel 1 (
            echo       insightface 설치 완료!
            del "!WHEEL_FILE!" 2>nul
        ) else (
            echo       [경고] insightface 휠 설치 실패. pip로 재시도...
            pip install insightface --prefer-binary -q 2>nul
        )
    ) else (
        echo       [경고] insightface 휠 다운로드 불완전. pip로 재시도...
        del "!WHEEL_FILE!" 2>nul
        pip install insightface --prefer-binary -q 2>nul
    )
) else (
    echo       [경고] insightface 휠 다운로드 실패. pip로 재시도...
    pip install insightface --prefer-binary -q 2>nul
)
echo       LivePortrait 의존성 설치 완료!

:lp_deps_done

REM Download LivePortrait model files
if not exist "models\live_portrait\liveportrait\base_models\appearance_feature_extractor.pth" (
    echo       LivePortrait 모델 파일 다운로드 중... (약 400MB)

    if not exist "models\live_portrait" mkdir "models\live_portrait"

    REM Download LivePortrait model from HuggingFace
    python -c "from huggingface_hub import snapshot_download; snapshot_download('KwaiVGI/LivePortrait', local_dir='models/live_portrait', local_dir_use_symlinks=False)"
    echo       LivePortrait 모델 다운로드 완료!
) else (
    echo       LivePortrait 모델 확인됨
)

REM Create symbolic link for LivePortrait pretrained_weights (required by LivePortrait pipeline)
REM LivePortrait expects models in external/LivePortrait/pretrained_weights/ but we download to models/live_portrait/
if not exist "external\LivePortrait\pretrained_weights" (
    echo       LivePortrait 모델 링크 생성 중...

    REM Get absolute path to models/live_portrait
    for %%i in ("models\live_portrait") do set "MODEL_PATH=%%~fi"

    REM Create junction (directory symbolic link on Windows)
    mklink /J "external\LivePortrait\pretrained_weights" "%MODEL_PATH%" >nul 2>&1
    if errorlevel 1 (
        echo       [참고] Junction 생성 실패. 폴더 복사로 대체합니다...
        xcopy "models\live_portrait\*" "external\LivePortrait\pretrained_weights\" /s /e /i /y /q >nul
    ) else (
        echo       LivePortrait 모델 링크 생성 완료!
    )
) else (
    echo       LivePortrait 모델 링크 확인됨
)

REM ============================================================
REM 7. Frontend package installation
REM ============================================================
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

REM ============================================================
REM 8. Environment configuration
REM ============================================================
echo.
echo [8/10] 환경 설정 확인 중...

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo       .env 파일 생성됨
    )
)

REM Change DEVICE to cuda in .env if GPU detected
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

REM ============================================================
REM 8.5 Force final NumPy compatibility (after all packages installed)
REM ============================================================
echo.
echo [8.5/10] NumPy 최종 호환성 강제 적용 중...
python -c "import numpy; v=numpy.__version__; exit(0 if int(v.split('.')[0]) < 2 else 1)" 2>nul
if errorlevel 1 (
    echo       [!] NumPy 2.x가 다시 설치됨, 최종 강제 다운그레이드 중...
    pip uninstall numpy -y >nul 2>&1
    pip cache purge >nul 2>&1
    pip install numpy==1.26.4 --no-cache-dir --force-reinstall -q 2>nul
    if errorlevel 1 (
        echo       [경고] 권한 오류 발생, --user 옵션으로 재시도 중...
        pip install numpy==1.26.4 --no-cache-dir --force-reinstall --user -q
    )

    REM Recompile dependent packages (NumPy header compatibility)
    echo       pandas/mediapipe 재설치 중...
    pip uninstall pandas mediapipe -y >nul 2>&1
    pip install pandas --no-cache-dir -q
    pip install mediapipe --no-cache-dir -q

    python -c "import numpy; print(f'       NumPy 버전: {numpy.__version__}')"
    echo       NumPy 호환성 강제 적용 완료!
) else (
    python -c "import numpy; print(f'       NumPy 버전: {numpy.__version__} [호환]')"
)

REM ============================================================
REM 9. Server startup (Backend + Frontend)
REM ============================================================
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

REM Set eSpeak-ng PATH (for Zonos TTS)
set "ESPEAK_PATH="
if exist "C:\Program Files\eSpeak NG\espeak-ng.exe" set "ESPEAK_PATH=C:\Program Files\eSpeak NG"
if exist "C:\Program Files (x86)\eSpeak NG\espeak-ng.exe" set "ESPEAK_PATH=C:\Program Files (x86)\eSpeak NG"

REM Start backend in new window (with eSpeak-ng PATH, torch.compile disabled)
REM TORCHDYNAMO_DISABLE=1: Prevents Triton warning on Windows
REM --log-level info: INFO 레벨 로그 활성화 (Python logging 설정과 호환)
if defined ESPEAK_PATH (
    start "Backend - AI Avatar" cmd /k "cd /d %~dp0 && set PATH=%ESPEAK_PATH%;%PATH% && set PHONEMIZER_ESPEAK_LIBRARY=%ESPEAK_PATH%\libespeak-ng.dll && set TORCHDYNAMO_DISABLE=1 && set PYTHONUNBUFFERED=1 && python -u -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --log-level info"
) else (
    echo       [경고] eSpeak-ng가 설치되지 않았습니다. Zonos TTS가 작동하지 않을 수 있습니다.
    start "Backend - AI Avatar" cmd /k "cd /d %~dp0 && set TORCHDYNAMO_DISABLE=1 && set PYTHONUNBUFFERED=1 && python -u -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --log-level info"
)

REM Wait for backend server to be ready
echo       백엔드 서버 초기화 대기 중...
:wait_backend
timeout /t 2 /nobreak >nul
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo       ... 백엔드 초기화 중 ...
    goto wait_backend
)
echo       백엔드 서버 준비 완료!

REM Start frontend in new window
start "Frontend - AI Avatar" cmd /k "cd /d %~dp0frontend && npm run dev"

REM Wait for frontend server to be ready
echo       프론트엔드 서버 초기화 대기 중...
:wait_frontend
timeout /t 2 /nobreak >nul
curl -s http://localhost:5173 >nul 2>&1
if errorlevel 1 (
    echo       ... 프론트엔드 초기화 중 ...
    goto wait_frontend
)
echo       프론트엔드 서버 준비 완료!

REM Open browser after all servers are ready
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
