#!/bin/bash
#
# AI Avatar 자동 실행 스크립트
# 실행만 하면 설치 → 체크 → 서버 실행까지 자동 진행
#
# Usage: ./start.sh
#

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
#                        로고
# ============================================================

print_logo() {
    clear
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║                                                           ║"
    echo "║     █████╗ ██╗     █████╗ ██╗   ██╗ █████╗ ████████╗     ║"
    echo "║    ██╔══██╗██║    ██╔══██╗██║   ██║██╔══██╗╚══██╔══╝     ║"
    echo "║    ███████║██║    ███████║██║   ██║███████║   ██║        ║"
    echo "║    ██╔══██║██║    ██╔══██║╚██╗ ██╔╝██╔══██║   ██║        ║"
    echo "║    ██║  ██║██║    ██║  ██║ ╚████╔╝ ██║  ██║   ██║        ║"
    echo "║    ╚═╝  ╚═╝╚═╝    ╚═╝  ╚═╝  ╚═══╝  ╚═╝  ╚═╝   ╚═╝        ║"
    echo "║                                                           ║"
    echo "║            Realtime AI Avatar Service                     ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  STEP $1: $2${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# ============================================================
#                   STEP 1: 의존성 설치
# ============================================================

install_dependencies() {
    print_step "1/4" "의존성 설치"

    # 이미 설치 확인
    if python3 -c "import torch, funasr, openai, livekit" 2>/dev/null; then
        echo -e "${GREEN}✅ 핵심 패키지 이미 설치됨 - 스킵${NC}"
        return 0
    fi

    echo "패키지 설치 중..."

    # GPU 체크
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        TORCH_URL="https://download.pytorch.org/whl/cu118"
        echo -e "${GREEN}GPU(CUDA) 감지${NC}"
    else
        TORCH_URL="https://download.pytorch.org/whl/cpu"
        echo -e "${YELLOW}CPU 모드${NC}"
    fi

    # PyTorch
    echo "  → PyTorch 설치..."
    pip install torch torchaudio torchvision --index-url $TORCH_URL -q 2>/dev/null || pip install torch torchaudio torchvision -q

    # 기본 패키지
    echo "  → 기본 패키지 설치..."
    pip install fastapi uvicorn python-dotenv websockets aiofiles pydantic numpy -q 2>/dev/null || true
    pip install opencv-python-headless mediapipe -q 2>/dev/null || true

    # LLM
    echo "  → LLM SDK 설치..."
    pip install openai anthropic -q 2>/dev/null || true

    # LiveKit
    echo "  → LiveKit 설치..."
    pip install livekit livekit-api -q 2>/dev/null || true

    # ML 패키지
    echo "  → ML 패키지 설치..."
    pip install transformers diffusers accelerate huggingface_hub -q 2>/dev/null || true

    # STT (FunASR)
    echo "  → STT 모듈 설치..."
    pip install funasr modelscope -q 2>/dev/null || true
    pip install omegaconf kaldiio hydra-core torch-complex --no-deps -q 2>/dev/null || true

    # TTS (Chatterbox)
    echo "  → TTS 모듈 설치..."
    pip install edge-tts gtts -q 2>/dev/null || true
    pip install chatterbox-tts resemble-perth conformer --no-deps -q 2>/dev/null || true

    echo -e "${GREEN}✅ 의존성 설치 완료${NC}"
}

# ============================================================
#                   STEP 2: 모델 다운로드
# ============================================================

download_models() {
    print_step "2/4" "모델 다운로드"

    # 모델 체크
    if [ -f "models/musetalk/musetalkV15/unet.pth" ] && [ -f "models/musetalk/sd-vae-ft-mse/config.json" ]; then
        echo -e "${GREEN}✅ 모델 파일 이미 존재 - 스킵${NC}"
        return 0
    fi

    echo "모델 다운로드 중..."

    python3 << 'PYMODELS'
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download

    # 디렉토리 생성
    Path("models/musetalk/musetalkV15").mkdir(parents=True, exist_ok=True)

    # VAE 다운로드
    vae_dir = Path("models/musetalk/sd-vae-ft-mse")
    if not (vae_dir / "config.json").exists():
        print("  → VAE 모델 다운로드...")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=str(vae_dir),
        )
        print("  ✅ VAE 완료")
    else:
        print("  ✅ VAE 이미 존재")

    # UNet 체크
    unet_path = Path("models/musetalk/musetalkV15/unet.pth")
    if unet_path.exists():
        print("  ✅ UNet 이미 존재")
    else:
        print("  ⚠️  UNet 모델은 수동 다운로드 필요")
        print("      → https://huggingface.co/TMElyralab/MuseTalk")

except Exception as e:
    print(f"  ⚠️ 모델 다운로드 중 오류: {e}")
PYMODELS

    echo -e "${GREEN}✅ 모델 다운로드 완료${NC}"
}

# ============================================================
#                   STEP 3: 환경 설정
# ============================================================

setup_environment() {
    print_step "3/4" "환경 설정"

    # 디렉토리 생성
    mkdir -p assets/idle_loops models/musetalk logs

    # .env 파일 생성
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo -e "${YELLOW}⚠️  .env 파일 생성됨${NC}"
            echo -e "${YELLOW}   API 키를 설정해주세요:${NC}"
            echo "   - OPENAI_API_KEY 또는 ANTHROPIC_API_KEY"
        fi
    else
        echo -e "${GREEN}✅ .env 파일 존재${NC}"
    fi

    # 환경 체크
    echo ""
    echo -e "${YELLOW}[모듈 상태]${NC}"
    python3 << 'PYCHECK'
modules = [
    ("funasr", "STT"),
    ("chatterbox", "TTS"),
    ("openai", "OpenAI"),
    ("anthropic", "Claude"),
    ("livekit", "LiveKit"),
    ("torch", "PyTorch"),
]
for mod, name in modules:
    try:
        __import__(mod)
        print(f"  ✅ {name}")
    except:
        print(f"  ❌ {name}")
PYCHECK

    echo ""
    echo -e "${YELLOW}[모델 상태]${NC}"
    [ -f "models/musetalk/musetalkV15/unet.pth" ] && echo "  ✅ MuseTalk UNet" || echo "  ❌ MuseTalk UNet"
    [ -f "models/musetalk/sd-vae-ft-mse/config.json" ] && echo "  ✅ MuseTalk VAE" || echo "  ❌ MuseTalk VAE"

    echo ""
    echo -e "${GREEN}✅ 환경 설정 완료${NC}"
}

# ============================================================
#                   STEP 4: 서버 실행
# ============================================================

run_server() {
    print_step "4/4" "서버 실행"

    # .env 로드
    if [ -f ".env" ]; then
        set -a
        source .env 2>/dev/null || true
        set +a
    fi

    HOST=${HOST:-0.0.0.0}
    PORT=${PORT:-8000}

    echo -e "  서버 주소: ${CYAN}http://$HOST:$PORT${NC}"
    echo -e "  API 문서:  ${CYAN}http://$HOST:$PORT/docs${NC}"
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  서버 시작! 종료하려면 Ctrl+C${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    uvicorn src.main:app --host $HOST --port $PORT --reload
}

# ============================================================
#                        메인 실행
# ============================================================

print_logo

echo -e "${GREEN}AI Avatar 서비스를 시작합니다...${NC}"
echo ""

# 순차 실행
install_dependencies
download_models
setup_environment
run_server
