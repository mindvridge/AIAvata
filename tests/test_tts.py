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
