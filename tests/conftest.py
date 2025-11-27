"""
Pytest configuration and fixtures.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end")
    config.addinivalue_line("markers", "integration: marks tests as integration")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables"""
    os.environ.setdefault("DEVICE", "cpu")
    os.environ.setdefault("ANTHROPIC_API_KEY", "test_key")
    os.environ.setdefault("LIVEKIT_API_KEY", "test_key")
    os.environ.setdefault("LIVEKIT_API_SECRET", "test_secret")
    os.environ.setdefault("LIVEKIT_URL", "ws://localhost:7880")


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing"""
    import numpy as np

    duration = 1.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    t = np.linspace(0, duration, samples)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_bytes(sample_audio):
    """Generate sample audio bytes"""
    import numpy as np
    return (sample_audio * 32767).astype(np.int16).tobytes()


@pytest.fixture
def sample_video_frame():
    """Generate sample video frame (512x512 RGB)"""
    import numpy as np
    return np.zeros((512, 512, 3), dtype=np.uint8)
