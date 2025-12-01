#!/bin/bash
# AI Avatar - 원클릭 실행
cd "$(dirname "$0")"

echo "🚀 AI Avatar 시작..."

# 1. 패키지 체크 & 설치
python3 -c "import torch,funasr,openai,livekit" 2>/dev/null || {
    echo "📦 패키지 설치 중..."
    pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu -q 2>/dev/null
    pip install fastapi uvicorn python-dotenv websockets aiofiles pydantic numpy opencv-python-headless -q 2>/dev/null
    pip install openai anthropic livekit livekit-api transformers diffusers huggingface_hub -q 2>/dev/null
    pip install funasr modelscope omegaconf kaldiio -q 2>/dev/null
    pip install edge-tts gtts chatterbox-tts resemble-perth -q 2>/dev/null
    echo "✅ 설치 완료"
}

# 2. 모델 체크 & 다운로드
[ -f "models/musetalk/sd-vae-ft-mse/config.json" ] || {
    echo "📥 모델 다운로드 중..."
    python3 -c "
from huggingface_hub import snapshot_download
from pathlib import Path
Path('models/musetalk').mkdir(parents=True,exist_ok=True)
snapshot_download('stabilityai/sd-vae-ft-mse',local_dir='models/musetalk/sd-vae-ft-mse')
print('✅ 모델 완료')
" 2>/dev/null
}

# 3. .env 생성
[ -f ".env" ] || cp .env.example .env 2>/dev/null

# 4. 서버 실행
echo "🌐 서버 시작: http://0.0.0.0:8000"
[ -f ".env" ] && export $(grep -v '^#' .env | xargs) 2>/dev/null
exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --reload
