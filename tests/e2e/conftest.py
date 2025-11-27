"""
E2E Test Configuration and Fixtures.

E2E 테스트용 설정 및 픽스처
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import AsyncGenerator, Generator

import pytest
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables before importing app modules
os.environ.setdefault("DEVICE", "cpu")
os.environ.setdefault("ANTHROPIC_API_KEY", "test_key_for_e2e")
os.environ.setdefault("OPENAI_API_KEY", "test_key_for_e2e")
os.environ.setdefault("LIVEKIT_API_KEY", "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", "secret")
os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def app():
    """Create FastAPI test app."""
    from src.main import app
    return app


@pytest.fixture(scope="module")
async def async_client(app) -> AsyncGenerator:
    """Create async HTTP client for testing."""
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_audio_16k() -> np.ndarray:
    """Generate sample 16kHz audio (1 second)."""
    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    # Generate 440Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_bytes(sample_audio_16k) -> bytes:
    """Generate sample audio as int16 bytes."""
    return (sample_audio_16k * 32767).astype(np.int16).tobytes()


@pytest.fixture
def sample_speech_audio() -> np.ndarray:
    """Generate more realistic speech-like audio."""
    duration = 2.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)

    # Mix of frequencies to simulate speech
    audio = (
        0.3 * np.sin(2 * np.pi * 150 * t) +  # Fundamental
        0.2 * np.sin(2 * np.pi * 300 * t) +  # Harmonic
        0.1 * np.sin(2 * np.pi * 450 * t) +  # Harmonic
        0.1 * np.random.randn(samples)        # Noise
    )

    # Apply envelope
    envelope = np.ones(samples)
    attack = int(0.1 * sample_rate)
    release = int(0.1 * sample_rate)
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-release:] = np.linspace(1, 0, release)

    audio = audio * envelope * 0.5
    return audio.astype(np.float32)


@pytest.fixture
def sample_image() -> np.ndarray:
    """Generate sample avatar image (512x512 RGB)."""
    # Create a simple face-like pattern
    image = np.zeros((512, 512, 3), dtype=np.uint8)

    # Background
    image[:, :] = [200, 180, 160]  # Skin-like color

    # Simple face circle
    center = (256, 256)
    for y in range(512):
        for x in range(512):
            dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
            if dist < 200:
                image[y, x] = [220, 190, 170]

    return image


@pytest.fixture
def websocket_messages():
    """Helper for WebSocket message formats."""
    return {
        "start_session": {
            "type": "start_session",
            "avatar_id": "default",
            "language": "ko"
        },
        "audio_chunk": {
            "type": "audio",
            "format": "int16",
            "sample_rate": 16000
        },
        "text_input": {
            "type": "text",
            "content": "안녕하세요"
        },
        "stop_session": {
            "type": "stop"
        }
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return {
        "text": "안녕하세요! 무엇을 도와드릴까요?",
        "emotion": "happy"
    }


@pytest.fixture
def performance_thresholds():
    """Performance thresholds for E2E tests."""
    return {
        "first_response_latency_ms": 800,  # Target: ≤800ms
        "frame_rate_fps": 30,              # Target: 30 FPS
        "audio_latency_ms": 100,           # Target: ≤100ms
        "api_response_time_ms": 200,       # API response time
    }
