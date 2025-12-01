"""
TTS Module using Chatterbox.

Chatterbox 기반 텍스트-음성 변환 모듈
특징:
- 200ms 미만 지연시간
- 스트리밍 출력 지원
- 음성 클로닝 지원
- MIT 라이선스 (상업적 사용 가능)
"""

import asyncio
import io
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np

from ..models.schemas import TTSChunk
from ..models.integrations import ChatterboxTTSModel

logger = logging.getLogger(__name__)


class TTSModule:
    """
    Chatterbox 기반 TTS 모듈

    Features:
    - 텍스트 → 음성 변환
    - 스트리밍 출력
    - 음성 클로닝 (참조 오디오 사용)
    - 다국어 지원
    """

    def __init__(
        self,
        voice_sample_path: Optional[str] = None,
        sample_rate: int = 24000,
        device: str = "cuda",
        voice_id: str = "default",
    ):
        """
        Initialize TTS Module.

        Args:
            voice_sample_path: 음성 클로닝용 참조 오디오 경로
            sample_rate: 출력 샘플레이트
            device: Compute device
            voice_id: 음성 ID
        """
        self.voice_sample_path = voice_sample_path
        self.sample_rate = sample_rate
        self.device = device
        self.voice_id = voice_id

        self._chatterbox_model: Optional[ChatterboxTTSModel] = None
        self._initialized = False

    async def initialize(self) -> None:
        """모델 초기화"""
        if self._initialized:
            return

        logger.info("Initializing Chatterbox TTS model...")

        try:
            # ChatterboxTTSModel 초기화
            self._chatterbox_model = ChatterboxTTSModel(
                device=self.device,
                sample_rate=self.sample_rate,
            )

            success = await self._chatterbox_model.initialize()

            if not success:
                logger.warning("Chatterbox TTS initialization returned False, using fallback")

            # 음성 샘플 로드 (있는 경우)
            if self.voice_sample_path and Path(self.voice_sample_path).exists():
                await self._chatterbox_model.load_voice(
                    voice_path=self.voice_sample_path,
                    voice_id=self.voice_id,
                )
                logger.info(f"Loaded voice sample from {self.voice_sample_path}")

            self._initialized = True
            logger.info("Chatterbox TTS model initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            # Allow mock operation on failure
            self._initialized = True

    def _ensure_initialized(self) -> None:
        """초기화 확인"""
        if not self._initialized:
            raise RuntimeError("TTS module not initialized. Call initialize() first.")

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        텍스트를 음성으로 변환 (동기식, 전체 반환)

        Args:
            text: 변환할 텍스트
            voice_id: 사용할 음성 ID

        Returns:
            오디오 데이터 (numpy array)
        """
        self._ensure_initialized()

        if not text.strip():
            return np.array([], dtype=np.float32)

        # 음성 ID 결정
        use_voice_id = voice_id or self.voice_id

        if self._chatterbox_model is None:
            # Mock audio for testing
            logger.warning(f"TTS model not loaded, using mock audio for text: '{text[:50]}...'")
            return self._generate_mock_audio(len(text))

        try:
            # Chatterbox TTS 추론
            logger.debug(f"Calling Chatterbox TTS synthesize: text='{text[:50]}...', voice_id={use_voice_id}")
            audio = await self._chatterbox_model.synthesize(
                text=text,
                voice_id=use_voice_id,
            )

            if audio is not None and len(audio) > 0:
                logger.debug(f"TTS synthesize completed: {len(audio)} samples")
            else:
                logger.warning("TTS synthesize returned empty audio")

            return audio

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
            logger.warning(f"Falling back to mock audio for text: '{text[:50]}...'")
            return self._generate_mock_audio(len(text))

    async def synthesize_stream(
        self,
        text: str,
        chunk_size: int = 4096,
        voice_id: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        텍스트를 음성으로 변환 (스트리밍)

        Args:
            text: 변환할 텍스트
            chunk_size: 청크 크기 (샘플 수)
            voice_id: 사용할 음성 ID

        Yields:
            TTSChunk: 오디오 청크
        """
        self._ensure_initialized()

        if not text.strip():
            yield TTSChunk(
                audio_data=b"",
                sample_rate=self.sample_rate,
                duration_ms=0,
                is_last=True,
            )
            return

        # ChatterboxTTSModel의 스트리밍 합성 사용
        use_voice_id = voice_id or self.voice_id

        if self._chatterbox_model:
            chunk_index = 0
            async for audio_chunk in self._chatterbox_model.synthesize_stream(
                text=text,
                voice_id=use_voice_id,
                chunk_size=chunk_size,
            ):
                chunk_bytes = self._audio_to_bytes(audio_chunk)
                duration_ms = len(audio_chunk) / self.sample_rate * 1000

                yield TTSChunk(
                    audio_data=chunk_bytes,
                    sample_rate=self.sample_rate,
                    duration_ms=duration_ms,
                    is_last=False,
                )
                chunk_index += 1

            # 마지막 빈 청크로 종료 표시
            yield TTSChunk(
                audio_data=b"",
                sample_rate=self.sample_rate,
                duration_ms=0,
                is_last=True,
            )
        else:
            # 폴백: 전체 합성 후 청크 분할
            audio = await self.synthesize(text, voice_id)

            for j in range(0, len(audio), chunk_size):
                chunk_audio = audio[j : j + chunk_size]
                chunk_bytes = self._audio_to_bytes(chunk_audio)

                duration_ms = len(chunk_audio) / self.sample_rate * 1000
                is_chunk_last = (j + chunk_size >= len(audio))

                yield TTSChunk(
                    audio_data=chunk_bytes,
                    sample_rate=self.sample_rate,
                    duration_ms=duration_ms,
                    is_last=is_chunk_last,
                )

                await asyncio.sleep(duration_ms / 1000 * 0.1)

    async def synthesize_stream_realtime(
        self,
        text_stream: AsyncGenerator[str, None],
        chunk_size: int = 4096,
        voice_id: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        LLM 스트리밍 출력을 실시간으로 TTS 변환

        Args:
            text_stream: 텍스트 청크 스트림 (LLM 출력)
            chunk_size: 오디오 청크 크기
            voice_id: 사용할 음성 ID

        Yields:
            TTSChunk: 오디오 청크
        """
        buffer = ""

        async for text_chunk in text_stream:
            buffer += text_chunk

            # 문장이 완성되면 즉시 합성
            sentences, remaining = self._extract_complete_sentences(buffer)
            buffer = remaining

            for sentence in sentences:
                async for audio_chunk in self.synthesize_stream(
                    sentence, chunk_size, voice_id
                ):
                    # 마지막 여부는 전체 스트림에서 결정되므로 False로 설정
                    audio_chunk.is_last = False
                    yield audio_chunk

        # 남은 버퍼 처리
        if buffer.strip():
            async for audio_chunk in self.synthesize_stream(
                buffer, chunk_size, voice_id
            ):
                yield audio_chunk

    def _split_into_sentences(self, text: str) -> list:
        """텍스트를 문장 단위로 분리"""
        import re

        # 한국어/영어 문장 종결 패턴
        pattern = r"(?<=[.!?。！？])\s+"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_complete_sentences(self, text: str) -> tuple:
        """
        완성된 문장 추출

        Returns:
            (완성된 문장 리스트, 남은 텍스트)
        """
        import re

        # 문장 종결 부호 위치 찾기
        pattern = r"[.!?。！？]"
        matches = list(re.finditer(pattern, text))

        if not matches:
            return [], text

        # 마지막 문장 종결 부호까지 추출
        last_match = matches[-1]
        complete_text = text[: last_match.end()]
        remaining = text[last_match.end() :].lstrip()

        sentences = self._split_into_sentences(complete_text)
        return sentences, remaining

    def _audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """numpy array를 bytes로 변환"""
        # 16-bit PCM으로 변환
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _generate_mock_audio(self, text_length: int) -> np.ndarray:
        """테스트용 mock 오디오 생성"""
        # 텍스트 길이에 비례한 무음 오디오 (테스트 신호 추가)
        duration_samples = int(text_length * 0.1 * self.sample_rate)
        if duration_samples == 0:
            duration_samples = int(0.5 * self.sample_rate)  # 최소 0.5초
        
        # 무음 대신 테스트 신호(사인파) 생성 (파동이 보이도록)
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples)
        frequency = 440  # A4 음
        audio = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        logger.debug(f"Generated mock audio: {duration_samples} samples ({duration_samples/self.sample_rate:.2f}s)")
        return audio

    async def cleanup(self) -> None:
        """리소스 정리"""
        if self._chatterbox_model is not None:
            await self._chatterbox_model.cleanup()
            self._chatterbox_model = None
        self._initialized = False
        logger.info("TTS module cleaned up")
