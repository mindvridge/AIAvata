"""
Tests for Pipeline Orchestrator.
"""

import asyncio
import os
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.pipeline.orchestrator import PipelineOrchestrator
from src.config import Settings
from src.models.emotion import Emotion
from src.models.schemas import PipelineState


@pytest.fixture
def mock_settings():
    """Mock settings fixture"""
    settings = Settings(
        device="cpu",
        anthropic_api_key="test_key",
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-20250514",
        system_prompt="Test prompt",
        idle_loops_dir="assets/idle_loops",
        target_fps=30,
        video_width=256,
        video_height=256,
        tts_sample_rate=24000,
        tts_chunk_size=4096,
    )
    return settings


@pytest.fixture
def pipeline(mock_settings):
    """Pipeline fixture"""
    return PipelineOrchestrator(mock_settings)


class TestPipelineOrchestrator:
    """Pipeline Orchestrator 테스트"""

    @pytest.mark.asyncio
    async def test_initialization(self, pipeline):
        """초기화 테스트"""
        with patch.object(pipeline.stt, "initialize", new_callable=AsyncMock):
            with patch.object(pipeline.llm, "initialize", new_callable=AsyncMock):
                with patch.object(pipeline.tts, "initialize", new_callable=AsyncMock):
                    with patch.object(pipeline.renderer, "initialize", new_callable=AsyncMock):
                        await pipeline.initialize()
                        assert pipeline._initialized is True

    def test_create_session(self, pipeline):
        """세션 생성 테스트"""
        session = pipeline.create_session(
            avatar_id="test_avatar",
            system_prompt="Custom prompt",
        )

        assert session is not None
        assert session.avatar_id == "test_avatar"
        assert session.current_emotion == Emotion.NEUTRAL
        assert session.pipeline_state == PipelineState.IDLE

    def test_get_session(self, pipeline):
        """세션 조회 테스트"""
        created = pipeline.create_session()
        retrieved = pipeline.get_session(created.session_id)

        assert retrieved is not None
        assert retrieved.session_id == created.session_id

    def test_get_nonexistent_session(self, pipeline):
        """존재하지 않는 세션 조회 테스트"""
        from uuid import uuid4
        result = pipeline.get_session(uuid4())
        assert result is None

    def test_delete_session(self, pipeline):
        """세션 삭제 테스트"""
        session = pipeline.create_session()
        result = pipeline.delete_session(session.session_id)

        assert result is True
        assert pipeline.get_session(session.session_id) is None

    def test_get_metrics(self, pipeline):
        """메트릭 조회 테스트"""
        metrics = pipeline.get_metrics()

        assert "total_requests" in metrics
        assert "avg_latency_ms" in metrics
        assert "last_latency_ms" in metrics

    @pytest.mark.asyncio
    async def test_cleanup(self, pipeline):
        """cleanup 테스트"""
        with patch.object(pipeline.stt, "initialize", new_callable=AsyncMock):
            with patch.object(pipeline.llm, "initialize", new_callable=AsyncMock):
                with patch.object(pipeline.tts, "initialize", new_callable=AsyncMock):
                    with patch.object(pipeline.renderer, "initialize", new_callable=AsyncMock):
                        await pipeline.initialize()

        with patch.object(pipeline.stt, "cleanup", new_callable=AsyncMock):
            with patch.object(pipeline.llm, "cleanup", new_callable=AsyncMock):
                with patch.object(pipeline.tts, "cleanup", new_callable=AsyncMock):
                    with patch.object(pipeline.renderer, "cleanup", new_callable=AsyncMock):
                        await pipeline.cleanup()
                        assert pipeline._initialized is False


class TestAudioProcessing:
    """오디오 처리 테스트"""

    def test_audio_to_numpy(self):
        """bytes → numpy 변환 테스트"""
        from src.utils.audio_utils import AudioProcessor

        processor = AudioProcessor()
        audio_bytes = np.zeros(1000, dtype=np.int16).tobytes()
        audio_array = processor.bytes_to_array(audio_bytes, "int16")

        assert isinstance(audio_array, np.ndarray)
        assert audio_array.dtype == np.int16
        assert len(audio_array) == 1000

    def test_audio_normalization(self):
        """오디오 정규화 테스트"""
        from src.utils.audio_utils import AudioProcessor

        processor = AudioProcessor()
        audio = np.random.randn(1000).astype(np.float32) * 10
        normalized = processor.normalize(audio)

        assert np.abs(normalized).max() <= 1.0


class TestVAD:
    """VAD 테스트"""

    def test_vad_initialization(self):
        """VAD 초기화 테스트"""
        from src.utils.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        result = vad.initialize()

        assert result is True

    def test_is_speech(self):
        """음성 감지 테스트"""
        from src.utils.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        vad.initialize()

        # 무음 테스트
        silent_audio = np.zeros(512, dtype=np.float32)
        is_speech, prob = vad.is_speech(silent_audio, return_probability=True)

        assert isinstance(is_speech, bool)
        assert 0 <= prob <= 1

    def test_get_state(self):
        """상태 조회 테스트"""
        from src.utils.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        state = vad.get_state()

        assert "is_speaking" in state
        assert "buffer_length_ms" in state
        assert "silence_duration_ms" in state
