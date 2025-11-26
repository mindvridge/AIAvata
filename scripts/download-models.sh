#!/bin/bash
# Model download script for AI Avatar Service
# Downloads required ML models from HuggingFace Hub
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}AI Avatar Service - Model Downloader${NC}"
echo "======================================="

# Model directories
MODELS_DIR="${PROJECT_DIR}/models"
HF_CACHE="${MODELS_DIR}/huggingface"
MUSETALK_DIR="${MODELS_DIR}/musetalk"
LIVEPORTRAIT_DIR="${MODELS_DIR}/liveportrait"

# Create directories
mkdir -p "$HF_CACHE" "$MUSETALK_DIR" "$LIVEPORTRAIT_DIR"

# Check for HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}Installing huggingface_hub...${NC}"
    pip install huggingface_hub
fi

# Parse arguments
MODELS_TO_DOWNLOAD=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            MODELS_TO_DOWNLOAD="all"
            shift
            ;;
        --stt)
            MODELS_TO_DOWNLOAD="${MODELS_TO_DOWNLOAD} stt"
            shift
            ;;
        --tts)
            MODELS_TO_DOWNLOAD="${MODELS_TO_DOWNLOAD} tts"
            shift
            ;;
        --lipsync)
            MODELS_TO_DOWNLOAD="${MODELS_TO_DOWNLOAD} lipsync"
            shift
            ;;
        --portrait)
            MODELS_TO_DOWNLOAD="${MODELS_TO_DOWNLOAD} portrait"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --all       Download all models"
            echo "  --stt       Download STT models (SenseVoice)"
            echo "  --tts       Download TTS models (Chatterbox)"
            echo "  --lipsync   Download lip sync models (MuseTalk)"
            echo "  --portrait  Download portrait models (LivePortrait)"
            echo "  -h, --help  Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --all                    # Download all models"
            echo "  $0 --stt --tts              # Download STT and TTS only"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Default to all if no specific models requested
if [[ -z "$MODELS_TO_DOWNLOAD" ]]; then
    MODELS_TO_DOWNLOAD="all"
fi

# Set HuggingFace cache directory
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"

echo ""
echo -e "${BLUE}Model cache directory: ${HF_CACHE}${NC}"
echo ""

# Download functions
download_stt_models() {
    echo -e "${GREEN}Downloading STT models (SenseVoice)...${NC}"

    python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

cache_dir = os.environ.get('HF_HOME', 'models/huggingface')

print("  Downloading SenseVoice-Small...")
try:
    snapshot_download(
        repo_id="FunAudioLLM/SenseVoiceSmall",
        cache_dir=cache_dir,
        local_dir_use_symlinks=False
    )
    print("  SenseVoice-Small downloaded successfully!")
except Exception as e:
    print(f"  Warning: Could not download SenseVoice-Small: {e}")
EOF

    echo -e "${GREEN}STT models download complete.${NC}"
}

download_tts_models() {
    echo -e "${GREEN}Downloading TTS models (Chatterbox)...${NC}"

    python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

cache_dir = os.environ.get('HF_HOME', 'models/huggingface')

print("  Downloading Chatterbox TTS...")
try:
    snapshot_download(
        repo_id="ResembleAI/chatterbox",
        cache_dir=cache_dir,
        local_dir_use_symlinks=False
    )
    print("  Chatterbox TTS downloaded successfully!")
except Exception as e:
    print(f"  Warning: Could not download Chatterbox TTS: {e}")
EOF

    echo -e "${GREEN}TTS models download complete.${NC}"
}

download_lipsync_models() {
    echo -e "${GREEN}Downloading Lip Sync models (MuseTalk)...${NC}"

    python3 << EOF
from huggingface_hub import snapshot_download
import os

musetalk_dir = "${MUSETALK_DIR}"

print("  Downloading MuseTalk models...")
try:
    snapshot_download(
        repo_id="TMElyralab/MuseTalk",
        local_dir=musetalk_dir,
        local_dir_use_symlinks=False
    )
    print("  MuseTalk downloaded successfully!")
except Exception as e:
    print(f"  Warning: Could not download MuseTalk: {e}")
EOF

    echo -e "${GREEN}Lip sync models download complete.${NC}"
}

download_portrait_models() {
    echo -e "${GREEN}Downloading Portrait Animation models (LivePortrait)...${NC}"

    python3 << EOF
from huggingface_hub import snapshot_download
import os

liveportrait_dir = "${LIVEPORTRAIT_DIR}"

print("  Downloading LivePortrait models...")
try:
    snapshot_download(
        repo_id="KwaiVGI/LivePortrait",
        local_dir=liveportrait_dir,
        local_dir_use_symlinks=False
    )
    print("  LivePortrait downloaded successfully!")
except Exception as e:
    print(f"  Warning: Could not download LivePortrait: {e}")
EOF

    echo -e "${GREEN}Portrait animation models download complete.${NC}"
}

# Run downloads based on selection
if [[ "$MODELS_TO_DOWNLOAD" == "all" ]]; then
    download_stt_models
    echo ""
    download_tts_models
    echo ""
    download_lipsync_models
    echo ""
    download_portrait_models
else
    if [[ "$MODELS_TO_DOWNLOAD" == *"stt"* ]]; then
        download_stt_models
        echo ""
    fi
    if [[ "$MODELS_TO_DOWNLOAD" == *"tts"* ]]; then
        download_tts_models
        echo ""
    fi
    if [[ "$MODELS_TO_DOWNLOAD" == *"lipsync"* ]]; then
        download_lipsync_models
        echo ""
    fi
    if [[ "$MODELS_TO_DOWNLOAD" == *"portrait"* ]]; then
        download_portrait_models
        echo ""
    fi
fi

echo ""
echo -e "${GREEN}Model download complete!${NC}"
echo ""
echo "Model locations:"
echo "  HuggingFace cache: ${HF_CACHE}"
echo "  MuseTalk: ${MUSETALK_DIR}"
echo "  LivePortrait: ${LIVEPORTRAIT_DIR}"
echo ""
echo -e "${BLUE}Disk usage:${NC}"
du -sh "${MODELS_DIR}"/* 2>/dev/null || echo "  (No models downloaded yet)"
