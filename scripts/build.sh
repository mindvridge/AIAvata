#!/bin/bash
# Build script for AI Avatar Service Docker images
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building AI Avatar Service Docker Image${NC}"
echo "=========================================="

# Default values
IMAGE_NAME="ai-avatar-service"
IMAGE_TAG="latest"
BUILD_TARGET="runtime"
NO_CACHE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --tag TAG       Docker image tag (default: latest)"
            echo "  --no-cache      Build without cache"
            echo "  --target TARGET Build target (default: runtime)"
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

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}Warning: NVIDIA GPU not detected. Image will still be built with CUDA support.${NC}"
fi

# Build the image
echo ""
echo -e "${GREEN}Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo "Target: ${BUILD_TARGET}"
echo ""

DOCKER_BUILDKIT=1 docker build \
    ${NO_CACHE} \
    --target "${BUILD_TARGET}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f Dockerfile \
    .

# Show image info
echo ""
echo -e "${GREEN}Build complete!${NC}"
echo ""
docker images "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo -e "${GREEN}To run the container:${NC}"
echo "  docker-compose up -d"
echo ""
echo -e "${GREEN}To run with GPU:${NC}"
echo "  docker run --gpus all -p 8000:8000 ${IMAGE_NAME}:${IMAGE_TAG}"
