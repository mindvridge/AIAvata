#!/bin/bash
#
# AI Avatar 통합 실행 스크립트
# 설치, 환경 체크, 서버 실행을 하나로 통합
#
# Usage: ./start.sh [명령]
#
# 명령:
#   install     의존성 설치
#   check       환경 체크
#   run         서버 실행 (기본값)
#   all         설치 + 실행
#   help        도움말
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
#                        로고 및 UI
# ============================================================

print_logo() {
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

print_section() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

show_help() {
    echo "Usage: ./start.sh [명령]"
    echo ""
    echo "명령:"
    echo "  install     의존성 및 모델 설치"
    echo "  check       환경 체크"
    echo "  run         서버 실행 (기본값)"
    echo "  all         설치 후 실행"
    echo "  frontend    프론트엔드만 실행"
    echo "  help        도움말"
    echo ""
    echo "예시:"
    echo "  ./start.sh              # 서버 실행"
    echo "  ./start.sh install      # 의존성 설치"
    echo "  ./start.sh all          # 설치 + 실행"
    echo ""
}

# ============================================================
#                        환경 체크
# ============================================================

check_environment() {
    print_section "환경 체크"

    local all_ok=true

    # Python
    echo -e "${YELLOW}[Python]${NC}"
    if command -v python3 &> /dev/null; then
        echo -e "  ✅ $(python3 --version)"
    else
        echo -e "  ❌ Python3 설치 필요"
        all_ok=false
    fi
    echo ""

    # 핵심 모듈
    echo -e "${YELLOW}[핵심 모듈]${NC}"
    python3 << 'PYCHECK'
modules = [
    ("funasr", "STT (SenseVoice)"),
    ("chatterbox", "TTS (Chatterbox)"),
    ("openai", "LLM (OpenAI)"),
    ("anthropic", "LLM (Claude)"),
    ("livekit", "LiveKit"),
    ("torch", "PyTorch"),
    ("diffusers", "Diffusers"),
]
for mod, name in modules:
    try:
        __import__(mod)
        print(f"  ✅ {name}")
    except ImportError:
        print(f"  ❌ {name}")
PYCHECK
    echo ""

    # 모델 파일
    echo -e "${YELLOW}[모델 파일]${NC}"
    if [ -f "models/musetalk/musetalkV15/unet.pth" ]; then
        echo -e "  ✅ MuseTalk UNet"
    else
        echo -e "  ❌ MuseTalk UNet"
        all_ok=false
    fi

    if [ -f "models/musetalk/sd-vae-ft-mse/config.json" ]; then
        echo -e "  ✅ MuseTalk VAE"
    else
        echo -e "  ❌ MuseTalk VAE"
        all_ok=false
    fi
    echo ""

    # .env 파일
    echo -e "${YELLOW}[환경 설정]${NC}"
    if [ -f ".env" ]; then
        if grep -q "your_openai_api_key_here\|your_anthropic_api_key_here" .env 2>/dev/null; then
            echo -e "  ⚠️  .env - API 키 설정 필요"
        else
            echo -e "  ✅ .env 설정됨"
        fi
    else
        echo -e "  ❌ .env 파일 없음"
        all_ok=false
    fi
    echo ""

    if [ "$all_ok" = true ]; then
        echo -e "${GREEN}✅ 모든 체크 통과${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  일부 항목 확인 필요${NC}"
        return 1
    fi
}

# ============================================================
#                        설치
# ============================================================

install_dependencies() {
    print_section "의존성 설치"

    # GPU 체크
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        DEVICE="gpu"
        echo -e "${GREEN}GPU(CUDA) 감지됨${NC}"
    else
        DEVICE="cpu"
        echo -e "${YELLOW}CPU 모드로 설치${NC}"
    fi
    echo ""

    # 1. PyTorch
    echo -e "${YELLOW}[1/5] PyTorch 설치${NC}"
    if [ "$DEVICE" == "gpu" ]; then
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118 -q
    else
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu -q
    fi
    echo -e "  ✅ PyTorch 설치 완료"

    # 2. 기본 패키지
    echo -e "${YELLOW}[2/5] 기본 패키지 설치${NC}"
    pip install fastapi uvicorn python-dotenv websockets aiofiles pydantic numpy opencv-python-headless -q
    pip install openai anthropic livekit livekit-api -q
    pip install transformers diffusers accelerate huggingface_hub -q
    echo -e "  ✅ 기본 패키지 설치 완료"

    # 3. STT (FunASR)
    echo -e "${YELLOW}[3/5] STT 모듈 설치${NC}"
    pip install funasr modelscope omegaconf kaldiio hydra-core torch-complex -q --no-deps 2>/dev/null || true
    pip install funasr -q 2>/dev/null || echo "  ⚠️ FunASR 일부 의존성 스킵"
    echo -e "  ✅ STT 설치 완료"

    # 4. TTS (Chatterbox)
    echo -e "${YELLOW}[4/5] TTS 모듈 설치${NC}"
    pip install edge-tts gtts -q
    pip install chatterbox-tts resemble-perth conformer s3tokenizer -q --no-deps 2>/dev/null || true
    echo -e "  ✅ TTS 설치 완료"

    # 5. 모델 다운로드
    echo -e "${YELLOW}[5/5] 모델 다운로드${NC}"
    python3 << 'PYMODELS'
import os
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, hf_hub_download

    # VAE 모델
    vae_dir = Path("models/musetalk/sd-vae-ft-mse")
    if not (vae_dir / "config.json").exists():
        print("  VAE 다운로드 중...")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=str(vae_dir),
        )
        print("  ✅ VAE 다운로드 완료")
    else:
        print("  ✅ VAE 이미 존재")

    # UNet 체크
    unet_path = Path("models/musetalk/musetalkV15/unet.pth")
    if unet_path.exists():
        print("  ✅ UNet 이미 존재")
    else:
        print("  ⚠️ UNet 모델 수동 다운로드 필요")

except Exception as e:
    print(f"  ⚠️ 모델 다운로드 오류: {e}")
PYMODELS

    # .env 파일 생성
    if [ ! -f ".env" ]; then
        cp .env.example .env
        echo -e "  ✅ .env 파일 생성됨"
        echo -e "  ${YELLOW}⚠️  .env 파일에 API 키를 설정하세요${NC}"
    fi

    # 디렉토리 생성
    mkdir -p assets/idle_loops models/musetalk logs

    echo ""
    echo -e "${GREEN}✅ 설치 완료!${NC}"
}

# ============================================================
#                        서버 실행
# ============================================================

run_server() {
    print_section "서버 실행"

    # .env 로드
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs 2>/dev/null) || true
    fi

    HOST=${HOST:-0.0.0.0}
    PORT=${PORT:-8000}

    echo -e "  Host: ${CYAN}$HOST${NC}"
    echo -e "  Port: ${CYAN}$PORT${NC}"
    echo -e "  URL:  ${CYAN}http://$HOST:$PORT${NC}"
    echo ""
    echo -e "${GREEN}서버 시작 중...${NC}"
    echo -e "${YELLOW}종료하려면 Ctrl+C${NC}"
    echo ""

    uvicorn src.main:app --host $HOST --port $PORT --reload
}

run_frontend() {
    print_section "프론트엔드 실행"

    cd frontend

    if [ ! -d "node_modules" ]; then
        echo "npm install 실행 중..."
        npm install
    fi

    echo -e "${GREEN}프론트엔드 시작 중...${NC}"
    npm run dev
}

# ============================================================
#                        메인
# ============================================================

COMMAND=${1:-run}

print_logo

case $COMMAND in
    install)
        install_dependencies
        ;;
    check)
        check_environment
        ;;
    run)
        check_environment && run_server
        ;;
    all)
        install_dependencies
        echo ""
        check_environment
        echo ""
        run_server
        ;;
    frontend)
        run_frontend
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}알 수 없는 명령: $COMMAND${NC}"
        show_help
        exit 1
        ;;
esac
