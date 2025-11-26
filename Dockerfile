# Realtime AI Avatar Service
# Multi-stage build for optimized image size
# Supports: MuseTalk (lip sync), LivePortrait (idle loops), Chatterbox (TTS)

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ML Model paths
ENV HF_HOME=/app/models/huggingface
ENV TORCH_HOME=/app/models/torch
ENV MUSETALK_MODEL_DIR=/app/models/musetalk
ENV LIVEPORTRAIT_MODEL_DIR=/app/models/liveportrait

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    git \
    git-lfs \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Stage 2: Builder for Python packages
FROM base AS builder

WORKDIR /build

# Install pip and build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install HuggingFace Hub for model downloads
RUN pip install --no-cache-dir huggingface_hub>=0.20.0

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Final runtime image
FROM base AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN groupadd -r avatar && useradd -r -g avatar avatar

# Copy application code
COPY --chown=avatar:avatar . .

# Create necessary directories with proper structure
RUN mkdir -p \
    assets/avatars \
    assets/idle_loops \
    assets/voice_samples \
    logs \
    models/huggingface \
    models/torch \
    models/musetalk \
    models/liveportrait \
    && chown -R avatar:avatar assets logs models

# Switch to non-root user
USER avatar

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables for runtime
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=4

# Default command (use --workers for production)
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative: with model preloading
# CMD ["python", "-m", "tools.setup_models", "&&", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
