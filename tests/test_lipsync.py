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
