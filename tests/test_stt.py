"""
Tests for STT Module.
"""

import asyncio
import numpy as np
import pytest

from src.pipeline.stt_module import STTModule
from src.models.emotion import Emotion


@pytest.fixture
def stt_module():
    """STT 모듈 fixture"""
    return STTModule(device="cpu")


@pytest.fixture
def test_audio():
    """테스트 오디오 생성"""
    duration = 2.0
    sample_rate = 16000
    samples = int(duration * sample_rate)
    # 간단한 사인파
    t = np.linspace(0, duration, samples)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    return audio.astype(np.float32)


class TestSTTModule:
    """STT 모듈 테스트"""

    @pytest.mark.asyncio
    async def test_initialization(self, stt_module):
        """초기화 테스트"""
        await stt_module.initialize()
        assert stt_module._initialized is True

    @pytest.mark.asyncio
    async def test_transcribe_returns_result(self, stt_module, test_audio):
        """transcribe가 결과를 반환하는지 테스트"""
        await stt_module.initialize()
        result = await stt_module.transcribe(test_audio)

        assert result is not None
        assert hasattr(result, "text")
        assert hasattr(result, "emotion")
        assert hasattr(result, "language")
        assert hasattr(result, "confidence")

    @pytest.mark.asyncio
    async def test_transcribe_emotion_is_valid(self, stt_module, test_audio):
        """반환된 감정이 유효한 값인지 테스트"""
        await stt_module.initialize()
        result = await stt_module.transcribe(test_audio)

        assert isinstance(result.emotion, Emotion)

    @pytest.mark.asyncio
    async def test_preprocess_audio_mono(self, stt_module):
        """스테레오 → 모노 변환 테스트"""
        # 스테레오 오디오 생성
        stereo_audio = np.random.randn(1000, 2).astype(np.float32)
        mono_audio = stt_module._preprocess_audio(stereo_audio, 16000)

        assert len(mono_audio.shape) == 1

    @pytest.mark.asyncio
    async def test_preprocess_audio_normalization(self, stt_module):
        """오디오 정규화 테스트"""
        audio = np.random.randn(1000).astype(np.float32) * 10
        normalized = stt_module._preprocess_audio(audio, 16000)

        assert np.abs(normalized).max() <= 1.0

    @pytest.mark.asyncio
    async def test_cleanup(self, stt_module):
        """cleanup 테스트"""
        await stt_module.initialize()
        await stt_module.cleanup()

        assert stt_module._initialized is False
