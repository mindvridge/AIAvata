#!/bin/bash
#
# AI Avatar 의존성 설치 스크립트
# Usage: ./setup.sh [옵션]
#
# 옵션:
#   --cpu       CPU 전용 설치
#   --gpu       GPU (CUDA) 설치
#   --models    모델만 다운로드
#   --frontend  프론트엔드만 설치
#   --all       전체 설치 (기본값)
#

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 프로젝트 루트
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 기본값
DEVICE="cpu"
MODELS_ONLY=false
FRONTEND_ONLY=false

# 로고
print_logo() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════╗"
    echo "║      AI Avatar Setup Script              ║"
    echo "╚══════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 도움말
show_help() {
    echo "Usage: ./setup.sh [옵션]"
    echo ""
    echo "옵션:"
    echo "  --cpu       CPU 전용 설치 (기본값)"
    echo "  --gpu       GPU (CUDA) 설치"
    echo "  --models    모델만 다운로드"
    echo "  --frontend  프론트엔드만 설치"
    echo "  --all       전체 설치"
    echo "  --help      도움말"
}

# Python 패키지 설치
install_python_packages() {
    echo -e "${YELLOW}[1/4] Python 패키지 설치${NC}"

    # 기본 패키지
    pip install --upgrade pip

    # PyTorch 설치
    if [ "$DEVICE" == "gpu" ]; then
        echo "  CUDA용 PyTorch 설치..."
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "  CPU용 PyTorch 설치..."
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
    fi

    # requirements.txt
    if [ -f "requirements.txt" ]; then
        echo "  requirements.txt 설치..."
        pip install -r requirements.txt
    fi

    # 추가 패키지
    echo "  추가 패키지 설치..."
    pip install openai anthropic livekit livekit-api
    pip install transformers diffusers accelerate
    pip install edge-tts gtts

    # FunASR (SenseVoice)
    echo "  FunASR 설치..."
    pip install funasr modelscope omegaconf kaldiio hydra-core torch-complex --no-deps 2>/dev/null || true

    # Chatterbox TTS
    echo "  Chatterbox TTS 설치..."
    pip install chatterbox-tts resemble-perth conformer --no-deps 2>/dev/null || true

    echo -e "${GREEN}  ✓ Python 패키지 설치 완료${NC}"
}

# 모델 다운로드
download_models() {
    echo -e "${YELLOW}[2/4] 모델 다운로드${NC}"

    python3 << 'EOF'
import os
from pathlib import Path

print("  MuseTalk 모델 다운로드...")

try:
    from huggingface_hub import snapshot_download, hf_hub_download

    # MuseTalk UNet
    unet_dir = Path("models/musetalk/musetalkV15")
    unet_dir.mkdir(parents=True, exist_ok=True)

    if not (unet_dir / "unet.pth").exists():
        print("    - UNet 다운로드 중...")
        hf_hub_download(
            repo_id="TMElyralab/MuseTalk",
            filename="models/musetalk/musetalk.json",
            local_dir="models/musetalk/musetalkV15",
            local_dir_use_symlinks=False
        )
        hf_hub_download(
            repo_id="TMElyralab/MuseTalk",
            filename="models/musetalk/pytorch_model.bin",
            local_dir="models/musetalk/musetalkV15",
            local_dir_use_symlinks=False
        )
        # Rename if needed
        src = unet_dir / "models/musetalk/pytorch_model.bin"
        if src.exists():
            src.rename(unet_dir / "unet.pth")
    else:
        print("    - UNet 이미 존재")

    # VAE
    vae_dir = Path("models/musetalk/sd-vae-ft-mse")
    if not vae_dir.exists():
        print("    - VAE 다운로드 중...")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=str(vae_dir),
        )
    else:
        print("    - VAE 이미 존재")

    print("  ✓ 모델 다운로드 완료")

except Exception as e:
    print(f"  ⚠ 모델 다운로드 실패: {e}")
    print("    수동 다운로드가 필요할 수 있습니다.")
EOF

    echo -e "${GREEN}  ✓ 모델 다운로드 완료${NC}"
}

# 프론트엔드 설치
install_frontend() {
    echo -e "${YELLOW}[3/4] 프론트엔드 설치${NC}"

    if ! command -v npm &> /dev/null; then
        echo -e "${RED}  ✗ npm이 설치되어 있지 않습니다.${NC}"
        echo "    Node.js를 먼저 설치해주세요: https://nodejs.org/"
        return 1
    fi

    cd frontend
    echo "  npm install 실행 중..."
    npm install
    cd ..

    echo -e "${GREEN}  ✓ 프론트엔드 설치 완료${NC}"
}

# 환경 설정
setup_environment() {
    echo -e "${YELLOW}[4/4] 환경 설정${NC}"

    # .env 파일 생성
    if [ ! -f ".env" ]; then
        cp .env.example .env
        echo "  .env 파일 생성됨"
        echo -e "${YELLOW}  ⚠ .env 파일에 API 키를 설정해주세요:${NC}"
        echo "    - OPENAI_API_KEY"
        echo "    - ANTHROPIC_API_KEY"
        echo "    - LIVEKIT_API_KEY (선택)"
    else
        echo "  .env 파일 이미 존재"
    fi

    # 디렉토리 생성
    mkdir -p assets/idle_loops
    mkdir -p models/musetalk
    mkdir -p logs

    echo -e "${GREEN}  ✓ 환경 설정 완료${NC}"
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --gpu)
            DEVICE="gpu"
            shift
            ;;
        --models)
            MODELS_ONLY=true
            shift
            ;;
        --frontend)
            FRONTEND_ONLY=true
            shift
            ;;
        --all)
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}알 수 없는 옵션: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 메인 실행
print_logo

if [ "$MODELS_ONLY" = true ]; then
    download_models
    exit 0
fi

if [ "$FRONTEND_ONLY" = true ]; then
    install_frontend
    exit 0
fi

echo "설치 모드: $DEVICE"
echo ""

install_python_packages
download_models
install_frontend
setup_environment

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║         설치 완료!                        ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""
echo "다음 단계:"
echo "  1. .env 파일에 API 키 설정"
echo "  2. ./run.sh 로 서버 실행"
echo ""
