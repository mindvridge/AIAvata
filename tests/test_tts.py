"""
Tests for TTS Module.
"""

import asyncio
import numpy as np
import pytest

from src.pipeline.tts_module import TTSModule


@pytest.fixture
def tts_module():
    """TTS 모듈 fixture"""
    return TTSModule(sample_rate=24000, device="cpu")


class TestTTSModule:
    """TTS 모듈 테스트"""

    @pytest.mark.asyncio
    async def test_initialization(self, tts_module):
        """초기화 테스트"""
        await tts_module.initialize()
        assert tts_module._initialized is True

    @pytest.mark.asyncio
    async def test_synthesize_returns_audio(self, tts_module):
        """synthesize가 오디오를 반환하는지 테스트"""
        await tts_module.initialize()
        audio = await tts_module.synthesize("안녕하세요")

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self, tts_module):
        """빈 텍스트 처리 테스트"""
        await tts_module.initialize()
        audio = await tts_module.synthesize("")

        assert isinstance(audio, np.ndarray)
        assert len(audio) == 0

    @pytest.mark.asyncio
    async def test_synthesize_stream_yields_chunks(self, tts_module):
        """스트리밍 합성이 청크를 yield하는지 테스트"""
        await tts_module.initialize()

        chunks = []
        async for chunk in tts_module.synthesize_stream("안녕하세요. 반갑습니다."):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(hasattr(c, "audio_data") for c in chunks)
        assert any(c.is_last for c in chunks)

    @pytest.mark.asyncio
    async def test_split_into_sentences(self, tts_module):
        """문장 분리 테스트"""
        text = "안녕하세요. 반갑습니다! 어떻게 도와드릴까요?"
        sentences = tts_module._split_into_sentences(text)

        assert len(sentences) == 3
        assert "안녕하세요" in sentences[0]
        assert "반갑습니다" in sentences[1]

    @pytest.mark.asyncio
    async def test_extract_complete_sentences(self, tts_module):
        """완성된 문장 추출 테스트"""
        text = "안녕하세요. 반갑습니다. 저는"
        sentences, remaining = tts_module._extract_complete_sentences(text)

        assert len(sentences) == 2
        assert remaining == "저는"

    @pytest.mark.asyncio
    async def test_cleanup(self, tts_module):
        """cleanup 테스트"""
        await tts_module.initialize()
        await tts_module.cleanup()

        assert tts_module._initialized is False

    def test_find_early_split_point_comma(self, tts_module):
        """쉼표에서 조기 분리 테스트"""
        text = "안녕하세요, 저는 AI입니다"
        split_point = tts_module._find_early_split_point(text)

        assert split_point > 0
        assert text[:split_point].endswith(",")

    def test_find_early_split_point_space(self, tts_module):
        """공백에서 조기 분리 테스트"""
        text = "안녕하세요 저는 AI입니다 반갑습니다"
        split_point = tts_module._find_early_split_point(text)

        assert split_point > 0
        # 공백 다음이어야 함

    def test_find_early_split_point_short_text(self, tts_module):
        """짧은 텍스트는 분리하지 않음"""
        text = "안녕"
        split_point = tts_module._find_early_split_point(text)

        assert split_point == 0

    @pytest.mark.asyncio
    async def test_synthesize_stream_realtime_early_synthesis(self, tts_module):
        """조기 합성 테스트"""
        await tts_module.initialize()

        # 긴 텍스트를 청크로 나누어 스트리밍
        async def text_generator():
            # 문장 종결 부호 없이 긴 텍스트
            yield "안녕하세요 저는 인공지능 "
            yield "어시스턴트입니다 반갑습니다."

        chunks = []
        async for chunk in tts_module.synthesize_stream_realtime(
            text_generator(),
            min_text_for_early_synthesis=15,
        ):
            chunks.append(chunk)

        # 청크가 생성되었는지 확인
        assert len(chunks) > 0


class TestTTSChunkOptimization:
    """TTS 청크 최적화 테스트"""

    @pytest.fixture
    def tts_module(self):
        return TTSModule(sample_rate=24000, device="cpu")

    def test_initial_chunk_size_parameter(self, tts_module):
        """초기 청크 크기 파라미터 테스트"""
        # synthesize_stream_realtime이 initial_chunk_size 파라미터를 받는지 확인
        import inspect
        sig = inspect.signature(tts_module.synthesize_stream_realtime)
        params = list(sig.parameters.keys())

        assert "initial_chunk_size" in params
        assert "min_text_for_early_synthesis" in params

    @pytest.mark.asyncio
    async def test_audio_to_bytes_format(self, tts_module):
        """오디오 bytes 변환 형식 테스트"""
        await tts_module.initialize()

        # 테스트용 오디오 생성
        audio = np.array([0.5, -0.5, 0.3], dtype=np.float32)
        audio_bytes = tts_module._audio_to_bytes(audio)

        # 16-bit PCM으로 변환되었는지 확인
        assert isinstance(audio_bytes, bytes)
        assert len(audio_bytes) == len(audio) * 2  # 16-bit = 2 bytes per sample
