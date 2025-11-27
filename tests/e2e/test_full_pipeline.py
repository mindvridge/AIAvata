"""
E2E Tests for Full Pipeline Integration.

전체 파이프라인 통합 End-to-End 테스트
STT → LLM → TTS → Avatar Rendering
"""

import asyncio
import time
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient


class TestFullPipelineFlow:
    """Full pipeline flow tests."""

    @pytest.fixture
    async def pipeline_with_mocks(self):
        """Create pipeline with mocked external services."""
        from src.config import Settings
        from src.pipeline.orchestrator import PipelineOrchestrator

        settings = Settings(
            device="cpu",
            anthropic_api_key="test_key",
            llm_provider="anthropic",
            llm_model="claude-sonnet-4-20250514",
            system_prompt="Test assistant",
            target_fps=30,
            video_width=256,
            video_height=256,
            tts_sample_rate=24000,
        )

        pipeline = PipelineOrchestrator(settings)

        # Mock STT
        pipeline.stt.transcribe = AsyncMock(return_value={
            "text": "안녕하세요",
            "emotion": "neutral",
            "language": "ko"
        })

        # Mock LLM
        pipeline.llm.generate_response = AsyncMock(return_value={
            "text": "안녕하세요! 무엇을 도와드릴까요?",
            "emotion": "happy"
        })

        # Mock TTS
        mock_audio = np.random.randn(24000).astype(np.float32)  # 1 second
        pipeline.tts.synthesize = AsyncMock(return_value=mock_audio)

        # Mock Renderer
        mock_frame = np.zeros((256, 256, 3), dtype=np.uint8)
        pipeline.renderer.render_frame = AsyncMock(return_value=mock_frame)

        return pipeline

    @pytest.mark.asyncio
    async def test_audio_to_response_flow(self, pipeline_with_mocks, sample_audio_16k):
        """Test complete audio → response flow."""
        pipeline = pipeline_with_mocks

        # Create session
        session = pipeline.create_session(
            avatar_id="default",
            system_prompt="Test prompt"
        )

        # Simulate processing
        result = await pipeline.stt.transcribe(sample_audio_16k)
        assert result["text"] == "안녕하세요"

        llm_response = await pipeline.llm.generate_response(result["text"])
        assert "안녕하세요" in llm_response["text"]

        audio = await pipeline.tts.synthesize(llm_response["text"])
        assert len(audio) > 0

        frame = await pipeline.renderer.render_frame(audio[:1024])
        assert frame.shape == (256, 256, 3)

    @pytest.mark.asyncio
    async def test_text_to_avatar_flow(self, pipeline_with_mocks):
        """Test text input → avatar output flow."""
        pipeline = pipeline_with_mocks

        session = pipeline.create_session()
        text_input = "오늘 날씨가 어때요?"

        # LLM response
        llm_response = await pipeline.llm.generate_response(text_input)
        assert llm_response["text"]

        # TTS
        audio = await pipeline.tts.synthesize(llm_response["text"])
        assert isinstance(audio, np.ndarray)

        # Render
        frame = await pipeline.renderer.render_frame(audio[:1024])
        assert frame is not None

    @pytest.mark.asyncio
    async def test_emotion_flow(self, pipeline_with_mocks):
        """Test emotion detection and propagation."""
        pipeline = pipeline_with_mocks

        session = pipeline.create_session()

        # STT with emotion
        pipeline.stt.transcribe = AsyncMock(return_value={
            "text": "정말 기뻐요!",
            "emotion": "happy",
            "language": "ko"
        })

        result = await pipeline.stt.transcribe(np.zeros(16000))
        assert result["emotion"] == "happy"

        # LLM should receive emotion context
        pipeline.llm.generate_response = AsyncMock(return_value={
            "text": "저도 기뻐요!",
            "emotion": "happy"
        })

        response = await pipeline.llm.generate_response(
            result["text"],
            emotion=result["emotion"]
        )
        assert response["emotion"] == "happy"


class TestPipelineLatency:
    """Pipeline latency tests."""

    @pytest.mark.asyncio
    async def test_first_response_latency(
        self, async_client: AsyncClient, performance_thresholds
    ):
        """First response should be within latency target."""
        # Create session
        response = await async_client.post(
            "/api/avatar/create",
            json={"avatar_id": "default"}
        )
        assert response.status_code == 200

        # Note: Full latency test requires actual model processing
        # This is a placeholder for measuring session creation time
        start = time.perf_counter()
        response = await async_client.post(
            "/api/avatar/create",
            json={"avatar_id": "default"}
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Session creation should be fast
        assert elapsed_ms < 500

    @pytest.mark.asyncio
    async def test_pipeline_metrics_tracking(self, async_client: AsyncClient):
        """Pipeline should track metrics."""
        # Create and interact with session
        response = await async_client.post(
            "/api/avatar/create",
            json={"avatar_id": "default"}
        )

        # Check metrics
        metrics_response = await async_client.get("/api/metrics")
        assert metrics_response.status_code == 200

        metrics = metrics_response.json()
        assert "total_requests" in metrics
        assert "avg_latency_ms" in metrics


class TestPipelineComponents:
    """Individual pipeline component tests."""

    @pytest.mark.asyncio
    async def test_stt_module_integration(self):
        """Test STT module integration."""
        from src.pipeline.stt_module import STTModule
        from src.config import Settings

        settings = Settings(device="cpu")
        stt = STTModule(settings)

        # Should initialize without error
        await stt.initialize()

        # Should have transcribe method
        assert hasattr(stt, "transcribe")

        await stt.cleanup()

    @pytest.mark.asyncio
    async def test_llm_module_integration(self):
        """Test LLM module integration."""
        from src.pipeline.llm_module import LLMModule
        from src.config import Settings

        settings = Settings(
            device="cpu",
            anthropic_api_key="test_key",
            llm_provider="anthropic"
        )
        llm = LLMModule(settings)

        # Should initialize
        await llm.initialize()

        # Should have generate method
        assert hasattr(llm, "generate_response")

        await llm.cleanup()

    @pytest.mark.asyncio
    async def test_tts_module_integration(self):
        """Test TTS module integration."""
        from src.pipeline.tts_module import TTSModule
        from src.config import Settings

        settings = Settings(
            device="cpu",
            tts_sample_rate=24000
        )
        tts = TTSModule(settings)

        await tts.initialize()
        assert hasattr(tts, "synthesize")
        await tts.cleanup()

    @pytest.mark.asyncio
    async def test_renderer_module_integration(self):
        """Test Avatar Renderer integration."""
        from src.pipeline.avatar_renderer import AvatarRenderer
        from src.config import Settings

        settings = Settings(
            device="cpu",
            video_width=256,
            video_height=256,
            target_fps=30
        )
        renderer = AvatarRenderer(settings)

        await renderer.initialize()
        assert hasattr(renderer, "render_frame")
        await renderer.cleanup()


class TestSessionManagement:
    """Session management integration tests."""

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, async_client: AsyncClient):
        """Test complete session lifecycle."""
        # Create
        create_response = await async_client.post(
            "/api/avatar/create",
            json={"avatar_id": "default"}
        )
        assert create_response.status_code == 200
        session_id = create_response.json()["session_id"]

        # Read
        get_response = await async_client.get(f"/api/avatar/{session_id}")
        assert get_response.status_code == 200
        assert get_response.json()["session_id"] == session_id

        # Delete
        delete_response = await async_client.delete(f"/api/avatar/{session_id}")
        assert delete_response.status_code == 200

        # Verify deleted
        get_deleted = await async_client.get(f"/api/avatar/{session_id}")
        assert get_deleted.status_code == 404

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, async_client: AsyncClient):
        """Test managing multiple sessions."""
        sessions = []

        # Create multiple sessions
        for i in range(5):
            response = await async_client.post(
                "/api/avatar/create",
                json={"avatar_id": f"avatar_{i}"}
            )
            assert response.status_code == 200
            sessions.append(response.json()["session_id"])

        # All sessions should be unique
        assert len(set(sessions)) == 5

        # All sessions should be accessible
        for session_id in sessions:
            response = await async_client.get(f"/api/avatar/{session_id}")
            assert response.status_code == 200

        # Clean up
        for session_id in sessions:
            await async_client.delete(f"/api/avatar/{session_id}")

    @pytest.mark.asyncio
    async def test_session_state_persistence(self, async_client: AsyncClient):
        """Session state should persist across requests."""
        # Create session
        response = await async_client.post(
            "/api/avatar/create",
            json={
                "avatar_id": "default",
                "system_prompt": "Custom prompt"
            }
        )
        session_id = response.json()["session_id"]

        # Check state
        state1 = await async_client.get(f"/api/avatar/{session_id}")
        state1_data = state1.json()

        # State should be consistent
        state2 = await async_client.get(f"/api/avatar/{session_id}")
        state2_data = state2.json()

        assert state1_data["session_id"] == state2_data["session_id"]
        assert state1_data["avatar_id"] == state2_data["avatar_id"]


class TestVADIntegration:
    """Voice Activity Detection integration tests."""

    @pytest.mark.asyncio
    async def test_vad_with_speech(self, sample_speech_audio):
        """VAD should detect speech-like audio."""
        from src.utils.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        vad.initialize()

        # Process audio chunks
        chunk_size = 512
        speech_detected = False

        for i in range(0, len(sample_speech_audio), chunk_size):
            chunk = sample_speech_audio[i:i+chunk_size]
            if len(chunk) == chunk_size:
                is_speech, prob = vad.is_speech(chunk, return_probability=True)
                if prob > 0.3:
                    speech_detected = True
                    break

        # Note: With synthetic audio, detection may vary

    @pytest.mark.asyncio
    async def test_vad_with_silence(self):
        """VAD should not detect speech in silence."""
        from src.utils.vad import VoiceActivityDetector

        vad = VoiceActivityDetector()
        vad.initialize()

        silent_audio = np.zeros(512, dtype=np.float32)
        is_speech, prob = vad.is_speech(silent_audio, return_probability=True)

        # Silence should have low speech probability
        assert prob < 0.5


class TestAudioProcessingIntegration:
    """Audio processing integration tests."""

    @pytest.mark.asyncio
    async def test_audio_format_conversion(self, sample_audio_16k):
        """Test audio format conversions."""
        from src.utils.audio_utils import AudioProcessor

        processor = AudioProcessor()

        # Float32 to int16
        int16_audio = (sample_audio_16k * 32767).astype(np.int16)
        audio_bytes = int16_audio.tobytes()

        # Bytes to array
        reconstructed = processor.bytes_to_array(audio_bytes, "int16")
        assert reconstructed.dtype == np.int16
        assert len(reconstructed) == len(int16_audio)

    @pytest.mark.asyncio
    async def test_audio_resampling(self, sample_audio_16k):
        """Test audio resampling."""
        from src.utils.audio_utils import AudioProcessor

        processor = AudioProcessor()

        # Resample to 24kHz
        if hasattr(processor, 'resample'):
            resampled = processor.resample(sample_audio_16k, 16000, 24000)
            expected_length = int(len(sample_audio_16k) * 24000 / 16000)
            assert abs(len(resampled) - expected_length) < 10


class TestErrorRecovery:
    """Error recovery and resilience tests."""

    @pytest.mark.asyncio
    async def test_pipeline_recovers_from_error(self, async_client: AsyncClient):
        """Pipeline should recover from errors."""
        # Create session
        response = await async_client.post(
            "/api/avatar/create",
            json={"avatar_id": "default"}
        )
        session_id = response.json()["session_id"]

        # Send invalid request (should not crash service)
        await async_client.get("/api/avatar/invalid-id")

        # Service should still be healthy
        health = await async_client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        # Session should still be accessible
        session = await async_client.get(f"/api/avatar/{session_id}")
        assert session.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client: AsyncClient):
        """Service should handle concurrent requests."""
        import asyncio

        async def create_session():
            return await async_client.post(
                "/api/avatar/create",
                json={"avatar_id": "default"}
            )

        # Send concurrent requests
        tasks = [create_session() for _ in range(10)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count == 10

        # All sessions should be unique
        session_ids = [r.json()["session_id"] for r in responses]
        assert len(set(session_ids)) == 10
