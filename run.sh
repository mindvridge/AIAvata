#!/bin/bash
#
# AI Avatar 서버 실행 스크립트
# Usage: ./run.sh [옵션]
#
# 옵션:
#   --dev       개발 모드 (자동 리로드)
#   --prod      프로덕션 모드
#   --check     환경 체크만 실행
#   --frontend  프론트엔드만 실행
#   --help      도움말 표시
#

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 프로젝트 루트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 기본값
MODE="dev"
CHECK_ONLY=false
FRONTEND_ONLY=false

# 로고 출력
print_logo() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════╗"
    echo "║        AI Avatar Service                 ║"
    echo "║     Realtime Conversational Avatar       ║"
    echo "╚══════════════════════════════════════════╝"
    echo -e "${NC}"
}

# 도움말
show_help() {
    echo "Usage: ./run.sh [옵션]"
    echo ""
    echo "옵션:"
    echo "  --dev       개발 모드 (자동 리로드, 기본값)"
    echo "  --prod      프로덕션 모드"
    echo "  --check     환경 체크만 실행"
    echo "  --frontend  프론트엔드만 실행"
    echo "  --help      도움말 표시"
    echo ""
    echo "예시:"
    echo "  ./run.sh              # 개발 모드로 백엔드 실행"
    echo "  ./run.sh --prod       # 프로덕션 모드로 실행"
    echo "  ./run.sh --check      # 환경 체크"
}

# 환경 체크
check_environment() {
    echo -e "${YELLOW}[환경 체크]${NC}"

    # Python 체크
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1)
        echo -e "  Python: ${GREEN}✓${NC} $PYTHON_VERSION"
    else
        echo -e "  Python: ${RED}✗ 설치 필요${NC}"
        return 1
    fi

    # .env 파일 체크
    if [ -f ".env" ]; then
        if grep -q "your_" .env 2>/dev/null; then
            echo -e "  .env: ${YELLOW}⚠ API 키 설정 필요${NC}"
        else
            echo -e "  .env: ${GREEN}✓${NC}"
        fi
    else
        echo -e "  .env: ${RED}✗ 파일 없음${NC}"
        echo -e "       ${YELLOW}→ cp .env.example .env 실행 후 API 키 설정${NC}"
        return 1
    fi

    # 핵심 모듈 체크
    echo ""
    echo -e "${YELLOW}[모듈 체크]${NC}"
    python3 -c "
import sys

modules = {
    'STT': 'funasr',
    'TTS': 'chatterbox',
    'LLM (OpenAI)': 'openai',
    'LLM (Claude)': 'anthropic',
    'LiveKit': 'livekit',
    'PyTorch': 'torch',
}

for name, module in modules.items():
    try:
        __import__(module)
        print(f'  {name}: \033[0;32m✓\033[0m')
    except ImportError:
        print(f'  {name}: \033[0;31m✗ 설치 필요\033[0m')
"

    # 모델 파일 체크
    echo ""
    echo -e "${YELLOW}[모델 체크]${NC}"

    if [ -f "models/musetalk/musetalkV15/unet.pth" ]; then
        echo -e "  MuseTalk UNet: ${GREEN}✓${NC}"
    else
        echo -e "  MuseTalk UNet: ${RED}✗${NC}"
    fi

    if [ -d "models/musetalk/sd-vae-ft-mse" ]; then
        echo -e "  MuseTalk VAE: ${GREEN}✓${NC}"
    else
        echo -e "  MuseTalk VAE: ${RED}✗${NC}"
    fi

    echo ""
}

# 백엔드 서버 실행
run_backend() {
    echo -e "${GREEN}[백엔드 서버 시작]${NC}"

    # 환경 변수 로드
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
    fi

    HOST=${HOST:-0.0.0.0}
    PORT=${PORT:-8000}

    echo "  Host: $HOST"
    echo "  Port: $PORT"
    echo "  Mode: $MODE"
    echo ""

    if [ "$MODE" == "dev" ]; then
        echo -e "${BLUE}개발 모드로 실행 (자동 리로드 활성화)${NC}"
        uvicorn src.main:app --reload --host $HOST --port $PORT
    else
        echo -e "${BLUE}프로덕션 모드로 실행${NC}"
        uvicorn src.main:app --host $HOST --port $PORT --workers 4
    fi
}

# 프론트엔드 실행
run_frontend() {
    echo -e "${GREEN}[프론트엔드 시작]${NC}"

    cd frontend

    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}node_modules 없음. npm install 실행 중...${NC}"
        npm install
    fi

    npm run dev
}

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            MODE="dev"
            shift
            ;;
        --prod)
            MODE="prod"
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --frontend)
            FRONTEND_ONLY=true
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

if [ "$CHECK_ONLY" = true ]; then
    check_environment
    exit 0
fi

if [ "$FRONTEND_ONLY" = true ]; then
    run_frontend
    exit 0
fi

check_environment
echo ""
run_backend
