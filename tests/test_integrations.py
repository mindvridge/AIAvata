"""
Integration Tests for Model Integrations.

MuseTalk, LivePortrait, ChatterboxTTS 모델 통합 테스트
"""

import asyncio
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.integrations import MuseTalkModel, LivePortraitModel, ChatterboxTTSModel
from src.models.emotion import Emotion


# ============================================================================
# MuseTalk Integration Tests
# ============================================================================

class TestMuseTalkModel:
    """MuseTalk 모델 통합 테스트"""

    @pytest.fixture
    def musetalk_model(self):
        """MuseTalk 모델 fixture"""
        return MuseTalkModel(device="cpu", fp16=False)

    @pytest.mark.asyncio
    async def test_initialization(self, musetalk_model):
        """초기화 테스트"""
        result = await musetalk_model.initialize()
        # 모델이 설치되어 있지 않아도 폴백으로 True 반환
        assert result is True

    @pytest.mark.asyncio
    async def test_process_frame_returns_frame(self, musetalk_model):
        """process_frame이 프레임을 반환하는지 테스트"""
        await musetalk_model.initialize()

        # 테스트 프레임 생성 (512x512 BGR)
        test_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # 테스트 오디오 생성
        test_audio = np.random.randn(800).astype(np.float32)

        result = await musetalk_model.process_frame(
            source_frame=test_frame,
            audio_chunk=test_audio,
            audio_sample_rate=24000,
        )

        assert isinstance(result, np.ndarray)
        assert result.shape == test_frame.shape
        assert result.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_process_frame_empty_audio(self, musetalk_model):
        """빈 오디오 처리 테스트"""
        await musetalk_model.initialize()

        test_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        empty_audio = np.array([], dtype=np.float32)

        result = await musetalk_model.process_frame(
            source_frame=test_frame,
            audio_chunk=empty_audio,
            audio_sample_rate=24000,
        )

        # 빈 오디오면 원본 프레임 반환
        assert np.array_equal(result, test_frame)

    @pytest.mark.asyncio
    async def test_cleanup(self, musetalk_model):
        """cleanup 테스트"""
        await musetalk_model.initialize()
        await musetalk_model.cleanup()

        assert musetalk_model._initialized is False


# ============================================================================
# LivePortrait Integration Tests
# ============================================================================

class TestLivePortraitModel:
    """LivePortrait 모델 통합 테스트"""

    @pytest.fixture
    def live_portrait_model(self):
        """LivePortrait 모델 fixture"""
        return LivePortraitModel(
            device="cpu",
            output_size=(512, 512),
            fp16=False,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, live_portrait_model):
        """초기화 테스트"""
        result = await live_portrait_model.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_extract_source_features(self, live_portrait_model):
        """소스 이미지 특징 추출 테스트"""
        await live_portrait_model.initialize()

        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        result = await live_portrait_model.extract_source_features(test_image)

        assert result is True
        assert live_portrait_model._source_image is not None

    @pytest.mark.asyncio
    async def test_generate_frame(self, live_portrait_model):
        """단일 프레임 생성 테스트"""
        await live_portrait_model.initialize()

        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        await live_portrait_model.extract_source_features(test_image)

        frame = await live_portrait_model.generate_frame(
            motion_params={"head_yaw": 0.1, "head_pitch": 0.05},
        )

        assert isinstance(frame, np.ndarray)
        assert frame.shape[:2] == (512, 512)

    @pytest.mark.asyncio
    async def test_generate_idle_sequence(self, live_portrait_model):
        """idle 시퀀스 생성 테스트"""
        await live_portrait_model.initialize()

        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        await live_portrait_model.extract_source_features(test_image)

        frames = await live_portrait_model.generate_idle_sequence(
            emotion="neutral",
            num_frames=10,
            motion_intensity=0.3,
        )

        assert isinstance(frames, list)
        assert len(frames) == 10
        assert all(isinstance(f, np.ndarray) for f in frames)

    @pytest.mark.asyncio
    async def test_generate_idle_sequence_all_emotions(self, live_portrait_model):
        """모든 감정의 idle 시퀀스 생성 테스트"""
        await live_portrait_model.initialize()

        test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        await live_portrait_model.extract_source_features(test_image)

        emotions = ["neutral", "happy", "sad", "listening", "thinking"]

        for emotion in emotions:
            frames = await live_portrait_model.generate_idle_sequence(
                emotion=emotion,
                num_frames=5,
                motion_intensity=0.3,
            )

            assert len(frames) == 5, f"Failed for emotion: {emotion}"

    @pytest.mark.asyncio
    async def test_generate_without_source_image(self, live_portrait_model):
        """소스 이미지 없이 생성 시도 테스트"""
        await live_portrait_model.initialize()

        frames = await live_portrait_model.generate_idle_sequence(
            emotion="neutral",
            num_frames=5,
        )

        # 소스 이미지 없으면 빈 리스트 반환
        assert frames == [] or frames is None or len(frames) == 0

    @pytest.mark.asyncio
    async def test_cleanup(self, live_portrait_model):
        """cleanup 테스트"""
        await live_portrait_model.initialize()
        await live_portrait_model.cleanup()

        assert live_portrait_model._initialized is False


# ============================================================================
# ChatterboxTTS Integration Tests
# ============================================================================

class TestChatterboxTTSModel:
    """Chatterbox TTS 모델 통합 테스트"""

    @pytest.fixture
    def chatterbox_model(self):
        """Chatterbox TTS 모델 fixture"""
        return ChatterboxTTSModel(device="cpu", sample_rate=24000)

    @pytest.mark.asyncio
    async def test_initialization(self, chatterbox_model):
        """초기화 테스트"""
        result = await chatterbox_model.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio(self, chatterbox_model):
        """synthesize가 오디오를 반환하는지 테스트"""
        await chatterbox_model.initialize()

        audio = await chatterbox_model.synthesize("안녕하세요")

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, chatterbox_model):
        """빈 텍스트 처리 테스트"""
        await chatterbox_model.initialize()

        audio = await chatterbox_model.synthesize("")

        assert isinstance(audio, np.ndarray)
        assert len(audio) == 0

    @pytest.mark.asyncio
    async def test_synthesize_korean_text(self, chatterbox_model):
        """한국어 텍스트 합성 테스트"""
        await chatterbox_model.initialize()

        audio = await chatterbox_model.synthesize("안녕하세요. 반갑습니다.")

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    @pytest.mark.asyncio
    async def test_synthesize_english_text(self, chatterbox_model):
        """영어 텍스트 합성 테스트"""
        await chatterbox_model.initialize()

        audio = await chatterbox_model.synthesize("Hello, how are you?")

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    @pytest.mark.asyncio
    async def test_synthesize_stream_yields_chunks(self, chatterbox_model):
        """스트리밍 합성이 청크를 yield하는지 테스트"""
        await chatterbox_model.initialize()

        chunks = []
        async for chunk in chatterbox_model.synthesize_stream("안녕하세요"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(isinstance(c, np.ndarray) for c in chunks)

    @pytest.mark.asyncio
    async def test_load_voice(self, chatterbox_model, tmp_path):
        """음성 로드 테스트"""
        await chatterbox_model.initialize()

        # 테스트용 음성 파일 생성
        import soundfile as sf

        test_audio = np.random.randn(24000).astype(np.float32) * 0.5
        voice_path = tmp_path / "test_voice.wav"
        sf.write(str(voice_path), test_audio, 24000)

        result = await chatterbox_model.load_voice(
            voice_path=str(voice_path),
            voice_id="test_voice",
        )

        assert result is True
        assert "test_voice" in chatterbox_model._voices

    @pytest.mark.asyncio
    async def test_cleanup(self, chatterbox_model):
        """cleanup 테스트"""
        await chatterbox_model.initialize()
        await chatterbox_model.cleanup()

        assert chatterbox_model._initialized is False


# ============================================================================
# Avatar Renderer Integration Tests
# ============================================================================

class TestAvatarRendererIntegration:
    """Avatar Renderer 통합 테스트"""

    @pytest.fixture
    def avatar_renderer(self):
        """Avatar Renderer fixture"""
        from src.pipeline.avatar_renderer import AvatarRenderer
        return AvatarRenderer(
            idle_loops_dir="assets/idle_loops",
            output_width=256,  # 테스트용 작은 크기
            output_height=256,
            target_fps=30,
            device="cpu",
            use_fp16=False,
        )

    @pytest.mark.asyncio
    async def test_initialization(self, avatar_renderer):
        """초기화 테스트"""
        await avatar_renderer.initialize()
        assert avatar_renderer._initialized is True

    @pytest.mark.asyncio
    async def test_get_idle_frame(self, avatar_renderer):
        """idle 프레임 가져오기 테스트"""
        await avatar_renderer.initialize()

        frame = avatar_renderer.get_idle_frame()

        assert isinstance(frame, np.ndarray)
        assert frame.shape[:2] == (256, 256)

    @pytest.mark.asyncio
    async def test_set_emotion(self, avatar_renderer):
        """감정 설정 테스트"""
        await avatar_renderer.initialize()

        avatar_renderer.set_emotion(Emotion.HAPPY)
        assert avatar_renderer.get_current_emotion() == Emotion.HAPPY

        avatar_renderer.set_emotion(Emotion.SAD)
        assert avatar_renderer.get_current_emotion() == Emotion.SAD

    @pytest.mark.asyncio
    async def test_render_idle_stream(self, avatar_renderer):
        """idle 스트림 렌더링 테스트"""
        await avatar_renderer.initialize()

        frames = []
        async for frame in avatar_renderer.render_idle_stream(duration=0.1):
            frames.append(frame)
            if len(frames) >= 3:
                break

        assert len(frames) >= 1
        assert all(hasattr(f, "data") for f in frames)
        assert all(hasattr(f, "width") for f in frames)

    @pytest.mark.asyncio
    async def test_cleanup(self, avatar_renderer):
        """cleanup 테스트"""
        await avatar_renderer.initialize()
        await avatar_renderer.cleanup()

        assert avatar_renderer._initialized is False


# ============================================================================
# Full Pipeline Integration Tests
# ============================================================================

class TestFullPipelineIntegration:
    """전체 파이프라인 통합 테스트"""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings fixture"""
        from src.config import Settings

        return Settings(
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

    @pytest.mark.asyncio
    async def test_pipeline_modules_initialize_independently(self, mock_settings):
        """각 파이프라인 모듈이 독립적으로 초기화되는지 테스트"""
        from src.pipeline.stt_module import STTModule
        from src.pipeline.tts_module import TTSModule
        from src.pipeline.avatar_renderer import AvatarRenderer

        # STT
        stt = STTModule(device="cpu")
        await stt.initialize()
        assert stt._initialized is True
        await stt.cleanup()

        # TTS
        tts = TTSModule(device="cpu", sample_rate=24000)
        await tts.initialize()
        assert tts._initialized is True
        await tts.cleanup()

        # Renderer
        renderer = AvatarRenderer(device="cpu", output_width=256, output_height=256)
        await renderer.initialize()
        assert renderer._initialized is True
        await renderer.cleanup()

    @pytest.mark.asyncio
    async def test_emotion_flow_through_pipeline(self, mock_settings):
        """감정이 파이프라인을 통해 전달되는지 테스트"""
        from src.pipeline.avatar_renderer import AvatarRenderer
        from src.models.emotion import Emotion

        renderer = AvatarRenderer(device="cpu", output_width=256, output_height=256)
        await renderer.initialize()

        # 감정 변경
        emotions_to_test = [
            Emotion.NEUTRAL,
            Emotion.HAPPY,
            Emotion.SAD,
            Emotion.LISTENING,
        ]

        for emotion in emotions_to_test:
            renderer.set_emotion(emotion)
            current = renderer.get_current_emotion()
            assert current == emotion, f"Expected {emotion}, got {current}"

        await renderer.cleanup()


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """성능 테스트"""

    @pytest.mark.asyncio
    async def test_avatar_renderer_fps(self):
        """Avatar Renderer FPS 테스트"""
        import time
        from src.pipeline.avatar_renderer import AvatarRenderer

        renderer = AvatarRenderer(device="cpu", output_width=256, output_height=256)
        await renderer.initialize()

        # 100 프레임 렌더링 시간 측정
        start = time.perf_counter()
        for _ in range(100):
            renderer.get_idle_frame()
        elapsed = time.perf_counter() - start

        fps = 100 / elapsed

        await renderer.cleanup()

        # 최소 10 FPS 이상 (CPU에서)
        assert fps >= 10, f"FPS too low: {fps:.1f}"

    @pytest.mark.asyncio
    async def test_tts_latency(self):
        """TTS 지연시간 테스트"""
        import time
        from src.pipeline.tts_module import TTSModule

        tts = TTSModule(device="cpu", sample_rate=24000)
        await tts.initialize()

        # 짧은 텍스트 합성 시간 측정
        start = time.perf_counter()
        audio = await tts.synthesize("안녕")
        elapsed = (time.perf_counter() - start) * 1000

        await tts.cleanup()

        # 1초 이내 (mock 모드에서)
        assert elapsed < 1000, f"TTS latency too high: {elapsed:.1f}ms"
