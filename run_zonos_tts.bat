@echo off
chcp 65001 > nul
setlocal EnableDelayedExpansion

title Zonos TTS Voice Manager

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║               Zonos TTS Voice Manager                       ║
echo ║         음성 복제 및 관리 시스템 (한국어 지원)                ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

:: ===== 설정 =====
set "VENV_DIR=venv"
set "PORT=8100"
set "HOST=127.0.0.1"

:: ===== Python 확인 =====
echo [1/5] Python 확인 중...
python --version > nul 2>&1
if errorlevel 1 (
    echo       [오류] Python이 설치되어 있지 않습니다.
    echo       https://www.python.org/downloads/ 에서 Python 3.10+ 설치 후 다시 실행하세요.
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo       Python %PYTHON_VERSION% 발견

:: ===== 가상환경 확인/생성 =====
echo [2/5] 가상환경 확인 중...
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo       가상환경이 없습니다. 생성 중...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo       [오류] 가상환경 생성 실패
        pause
        exit /b 1
    )
    echo       가상환경 생성 완료
) else (
    echo       가상환경 발견
)

:: ===== 가상환경 활성화 =====
echo [3/5] 가상환경 활성화...
call %VENV_DIR%\Scripts\activate.bat
if errorlevel 1 (
    echo       [오류] 가상환경 활성화 실패
    pause
    exit /b 1
)
echo       활성화 완료

:: ===== 필수 패키지 설치 =====
echo [4/5] 필수 패키지 확인 및 설치...

:: FastAPI 확인
python -c "import fastapi" > nul 2>&1
if errorlevel 1 (
    echo       FastAPI 설치 중...
    pip install fastapi uvicorn python-multipart aiofiles --quiet
)

:: PyDub 확인 (오디오 처리)
python -c "import pydub" > nul 2>&1
if errorlevel 1 (
    echo       PyDub 설치 중...
    pip install pydub --quiet
)

:: NumPy 확인
python -c "import numpy" > nul 2>&1
if errorlevel 1 (
    echo       NumPy 설치 중...
    pip install numpy --quiet
)

:: Zonos 설치 시도
python -c "import zonos" > nul 2>&1
if errorlevel 1 (
    echo       Zonos TTS 설치 시도 중...
    pip install zonos --quiet 2>nul
    if errorlevel 1 (
        echo       [알림] Zonos 패키지 설치 실패 - Mock 모드로 실행됩니다.
        echo       실제 음성 합성을 위해서는 수동 설치가 필요합니다:
        echo       pip install zonos
    )
)

:: PyTorch 확인 (Zonos 의존성)
python -c "import torch" > nul 2>&1
if errorlevel 1 (
    echo       PyTorch 설치 중... (시간이 걸릴 수 있습니다)
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet 2>nul
    if errorlevel 1 (
        pip install torch torchaudio --quiet
    )
)

echo       패키지 확인 완료

:: ===== 음성 디렉토리 생성 =====
if not exist "assets\voices\audio" mkdir "assets\voices\audio"
if not exist "assets\voices\embeddings" mkdir "assets\voices\embeddings"

:: ===== 서버 실행 =====
echo [5/5] Zonos TTS Voice Manager 시작...
echo.
echo ══════════════════════════════════════════════════════════════
echo   서버 주소: http://%HOST%:%PORT%
echo   관리 페이지: http://%HOST%:%PORT%/
echo   API 문서: http://%HOST%:%PORT%/docs
echo ══════════════════════════════════════════════════════════════
echo.
echo   종료하려면 Ctrl+C 를 누르세요.
echo.

:: 브라우저 자동 열기 (3초 후)
start "" cmd /c "timeout /t 3 /nobreak > nul && start http://%HOST%:%PORT%/"

:: 서버 실행
python scripts/zonos_server.py --host %HOST% --port %PORT%

:: 에러 발생 시
if errorlevel 1 (
    echo.
    echo [오류] 서버 실행 중 오류가 발생했습니다.
    echo 로그를 확인해 주세요.
    pause
)

endlocal
