"""
Tests for Avatar Renderer (Lip Sync) Module.
"""

import asyncio
import numpy as np
import pytest

from src.pipeline.avatar_renderer import AvatarRenderer
from src.models.emotion import Emotion


@pytest.fixture
def avatar_renderer():
    """Avatar Renderer fixture"""
    return AvatarRenderer(
        output_width=256,
        output_height=256,
        target_fps=30,
        device="cpu",
        use_fp16=False,
    )


class TestAvatarRenderer:
    """Avatar Renderer 테스트"""

    @pytest.mark.asyncio
    async def test_initialization(self, avatar_renderer):
        """초기화 테스트"""
        await avatar_renderer.initialize()
        assert avatar_renderer._initialized is True

    @pytest.mark.asyncio
    async def test_get_idle_frame(self, avatar_renderer):
        """idle 프레임 반환 테스트"""
        await avatar_renderer.initialize()
        frame = avatar_renderer.get_idle_frame()

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (256, 256, 3)
        assert frame.dtype == np.uint8

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
        """idle 스트림 테스트"""
        await avatar_renderer.initialize()

        frames = []
        async for frame in avatar_renderer.render_idle_stream(duration=0.2):
            frames.append(frame)
            if len(frames) >= 5:
                break

        assert len(frames) >= 5
        assert all(hasattr(f, "data") for f in frames)
        assert all(hasattr(f, "width") for f in frames)
        assert all(hasattr(f, "height") for f in frames)

    @pytest.mark.asyncio
    async def test_render_with_audio(self, avatar_renderer):
        """오디오와 함께 렌더링 테스트"""
        await avatar_renderer.initialize()

        # 테스트 오디오 스트림
        async def audio_stream():
            for _ in range(3):
                # 24kHz 샘플레이트로 약 33ms (1프레임) 분량
                yield np.zeros(800, dtype=np.int16).tobytes()

        frames = []
        async for frame in avatar_renderer.render_with_audio(
            audio_stream(),
            audio_sample_rate=24000,
        ):
            frames.append(frame)

        assert len(frames) > 0

    @pytest.mark.asyncio
    async def test_cleanup(self, avatar_renderer):
        """cleanup 테스트"""
        await avatar_renderer.initialize()
        await avatar_renderer.cleanup()

        assert avatar_renderer._initialized is False


class TestEmotionMapping:
    """감정 매핑 테스트"""

    def test_sensevoice_mapping(self):
        """SenseVoice 감정 ID 매핑 테스트"""
        from src.models.emotion import EmotionMapping

        assert EmotionMapping.from_sensevoice(0) == Emotion.ANGRY
        assert EmotionMapping.from_sensevoice(1) == Emotion.HAPPY
        assert EmotionMapping.from_sensevoice(2) == Emotion.NEUTRAL
        assert EmotionMapping.from_sensevoice(3) == Emotion.SAD
        assert EmotionMapping.from_sensevoice(99) == Emotion.NEUTRAL  # 알 수 없는 ID

    def test_user_to_avatar_mapping(self):
        """사용자 감정 → 아바타 반응 매핑 테스트"""
        from src.models.emotion import EmotionMapping

        assert EmotionMapping.get_avatar_response(Emotion.HAPPY) == Emotion.HAPPY
        assert EmotionMapping.get_avatar_response(Emotion.SAD) == Emotion.SYMPATHETIC
        assert EmotionMapping.get_avatar_response(Emotion.ANGRY) == Emotion.CONCERNED

    def test_idle_loop_filename(self):
        """Idle 루프 파일명 매핑 테스트"""
        from src.models.emotion import EmotionMapping

        assert EmotionMapping.get_idle_loop_filename(Emotion.NEUTRAL) == "neutral_idle.mp4"
        assert EmotionMapping.get_idle_loop_filename(Emotion.HAPPY) == "happy_smile.mp4"


class TestAvatarRendererWithSourceImage:
    """소스 이미지를 사용하는 Avatar Renderer 테스트"""

    @pytest.fixture
    def renderer_with_image(self, tmp_path):
        """소스 이미지가 있는 Avatar Renderer fixture"""
        import cv2

        # 테스트 이미지 생성
        test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        image_path = tmp_path / "test_avatar.png"
        cv2.imwrite(str(image_path), test_image)

        return AvatarRenderer(
            idle_loops_dir=str(tmp_path / "idle_loops"),
            avatar_image_path=str(image_path),
            output_width=256,
            output_height=256,
            target_fps=30,
            device="cpu",
            use_fp16=False,
        )

    @pytest.mark.asyncio
    async def test_initialization_with_source_image(self, renderer_with_image):
        """소스 이미지와 함께 초기화 테스트"""
        await renderer_with_image.initialize()

        assert renderer_with_image._initialized is True
        assert renderer_with_image._source_image is not None

    @pytest.mark.asyncio
    async def test_frame_generation_with_source_image(self, renderer_with_image):
        """소스 이미지로 프레임 생성 테스트"""
        await renderer_with_image.initialize()

        frame = renderer_with_image.get_idle_frame()

        assert isinstance(frame, np.ndarray)
        assert frame.shape == (256, 256, 3)

    @pytest.mark.asyncio
    async def test_lipsync_applies_to_frame(self, renderer_with_image):
        """립싱크가 프레임에 적용되는지 테스트"""
        await renderer_with_image.initialize()

        # 원본 프레임
        original_frame = renderer_with_image.get_idle_frame().copy()

        # 오디오로 립싱크 적용
        test_audio = np.random.randn(800).astype(np.float32) * 0.5
        audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()

        lipsync_frame = await renderer_with_image._apply_lipsync(
            original_frame,
            audio_bytes,
        )

        assert isinstance(lipsync_frame, np.ndarray)
        assert lipsync_frame.shape == original_frame.shape


class TestLipSyncPerformance:
    """립싱크 성능 테스트"""

    @pytest.mark.asyncio
    async def test_frame_rate_target(self):
        """목표 프레임레이트 달성 테스트"""
        import time

        renderer = AvatarRenderer(
            output_width=256,
            output_height=256,
            target_fps=30,
            device="cpu",
            use_fp16=False,
        )
        await renderer.initialize()

        # 30프레임 렌더링 시간 측정
        start = time.perf_counter()
        for _ in range(30):
            renderer.get_idle_frame()
        elapsed = time.perf_counter() - start

        fps = 30 / elapsed

        await renderer.cleanup()

        # CPU에서 최소 15 FPS (30의 절반)
        assert fps >= 15, f"FPS too low for real-time: {fps:.1f}"

    @pytest.mark.asyncio
    async def test_emotion_switch_latency(self):
        """감정 전환 지연시간 테스트"""
        import time

        renderer = AvatarRenderer(
            output_width=256,
            output_height=256,
            target_fps=30,
            device="cpu",
            use_fp16=False,
        )
        await renderer.initialize()

        emotions = [Emotion.NEUTRAL, Emotion.HAPPY, Emotion.SAD, Emotion.LISTENING]

        for emotion in emotions:
            start = time.perf_counter()
            renderer.set_emotion(emotion)
            renderer.get_idle_frame()  # 감정 변경 후 첫 프레임
            elapsed = (time.perf_counter() - start) * 1000

            # 감정 전환은 100ms 이내
            assert elapsed < 100, f"Emotion switch too slow: {elapsed:.1f}ms"

        await renderer.cleanup()
