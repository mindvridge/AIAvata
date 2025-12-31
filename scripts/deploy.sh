#!/bin/bash
# Deployment script for AI Avatar Service
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}AI Avatar Service Deployment${NC}"
echo "=============================="

# Default values
ENVIRONMENT="development"
COMPOSE_FILE="docker-compose.yml"
DETACH="-d"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --production)
            ENVIRONMENT="production"
            shift
            ;;
        --cpu)
            COMPOSE_FILE="docker-compose.cpu.yml"
            shift
            ;;
        --foreground)
            DETACH=""
            shift
            ;;
        --down)
            cd "$PROJECT_DIR"
            echo -e "${YELLOW}Stopping services...${NC}"
            docker-compose -f "$COMPOSE_FILE" down
            echo -e "${GREEN}Services stopped.${NC}"
            exit 0
            ;;
        --logs)
            cd "$PROJECT_DIR"
            docker-compose -f "$COMPOSE_FILE" logs -f
            exit 0
            ;;
        --status)
            cd "$PROJECT_DIR"
            docker-compose -f "$COMPOSE_FILE" ps
            exit 0
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --env ENV       Deployment environment (development|production)"
            echo "  --production    Shortcut for --env production"
            echo "  --cpu           Use CPU-only configuration"
            echo "  --foreground    Run in foreground (don't detach)"
            echo "  --down          Stop all services"
            echo "  --logs          Follow service logs"
            echo "  --status        Show service status"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi
echo "  Docker: $(docker --version)"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    exit 1
fi
echo "  Docker Compose: $(docker-compose --version)"

# Check for NVIDIA GPU (for GPU deployment)
if [[ "$COMPOSE_FILE" != *"cpu"* ]]; then
    if command -v nvidia-smi &> /dev/null; then
        echo -e "  ${GREEN}NVIDIA GPU: Available${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    /'
    else
        echo -e "  ${YELLOW}NVIDIA GPU: Not detected (consider using --cpu flag)${NC}"
    fi
fi

# Check .env file
if [[ ! -f ".env" ]]; then
    echo -e "${YELLOW}Warning: .env file not found. Creating from template...${NC}"
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        echo -e "${YELLOW}Please edit .env file with your configuration${NC}"
    else
        cat > .env << 'EOF'
# AI Avatar Service Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514

# LiveKit Configuration
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
LIVEKIT_URL=ws://livekit:7880

# Performance Settings
TARGET_FPS=30
VIDEO_WIDTH=784
VIDEO_HEIGHT=1176
TTS_SAMPLE_RATE=24000

# Logging
LOG_LEVEL=INFO
EOF
        echo -e "${YELLOW}Created default .env file. Please edit with your API keys.${NC}"
    fi
fi

echo ""
echo -e "${BLUE}Environment: ${ENVIRONMENT}${NC}"
echo -e "${BLUE}Compose file: ${COMPOSE_FILE}${NC}"
echo ""

# Build images
echo -e "${GREEN}Building Docker images...${NC}"
docker-compose -f "$COMPOSE_FILE" build

# Create necessary directories
echo -e "${GREEN}Creating directories...${NC}"
mkdir -p assets/avatars assets/idle_loops assets/voice_samples logs models

# Add production profile if needed
PROFILES=""
if [[ "$ENVIRONMENT" == "production" ]]; then
    PROFILES="--profile production"
fi

# Start services
echo ""
echo -e "${GREEN}Starting services...${NC}"
docker-compose -f "$COMPOSE_FILE" $PROFILES up $DETACH

if [[ -n "$DETACH" ]]; then
    echo ""
    echo -e "${GREEN}Services started successfully!${NC}"
    echo ""
    echo "Useful commands:"
    echo "  View logs:     docker-compose logs -f"
    echo "  Stop services: docker-compose down"
    echo "  Service status: docker-compose ps"
    echo ""
    echo "API endpoints:"
    echo "  Health check:  http://localhost:8000/health"
    echo "  API docs:      http://localhost:8000/docs"
    echo "  WebSocket:     ws://localhost:8000/ws/"
fi
