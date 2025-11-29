#!/bin/bash
# MuseTalk VAE 모델 다운로드 스크립트
# sd-vae-ft-mse 모델 다운로드 (약 335MB)

set -e

MODELS_DIR="${1:-models}"
VAE_DIR="$MODELS_DIR/sd-vae-ft-mse"

echo "================================================"
echo "MuseTalk VAE (sd-vae-ft-mse) Download Script"
echo "================================================"

# 디렉토리 생성
mkdir -p "$VAE_DIR"

# 이미 다운로드됐는지 확인
if [ -f "$VAE_DIR/diffusion_pytorch_model.bin" ] && [ -f "$VAE_DIR/config.json" ]; then
    echo "✅ VAE 모델이 이미 존재합니다: $VAE_DIR"
    ls -lh "$VAE_DIR"
    exit 0
fi

echo "📥 VAE 모델 다운로드 중..."

# huggingface-cli 설치 확인
if ! command -v huggingface-cli &> /dev/null; then
    echo "huggingface-cli 설치 중..."
    pip install -U "huggingface_hub[cli]" -q
fi

# 다운로드
huggingface-cli download stabilityai/sd-vae-ft-mse \
    --local-dir "$VAE_DIR" \
    --include "config.json" "diffusion_pytorch_model.bin" "diffusion_pytorch_model.safetensors"

echo ""
echo "✅ VAE 모델 다운로드 완료!"
ls -lh "$VAE_DIR"
