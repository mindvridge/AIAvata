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
from typing import AsyncGenerator, Optional, Union

import numpy as np

from ..models.schemas import TTSChunk

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
    ):
        """
        Initialize TTS Module.

        Args:
            voice_sample_path: 음성 클로닝용 참조 오디오 경로
            sample_rate: 출력 샘플레이트
            device: Compute device
        """
        self.voice_sample_path = voice_sample_path
        self.sample_rate = sample_rate
        self.device = device

        self._model = None
        self._voice_sample = None
        self._initialized = False

    async def initialize(self) -> None:
        """모델 초기화"""
        if self._initialized:
            return

        logger.info("Initializing Chatterbox TTS model...")

        try:
            # Chatterbox TTS 임포트 및 초기화
            from chatterbox.tts import ChatterboxTTS

            self._model = ChatterboxTTS.from_pretrained(device=self.device)

            # 음성 샘플 로드 (있는 경우)
            if self.voice_sample_path and Path(self.voice_sample_path).exists():
                self._voice_sample = self._load_voice_sample(self.voice_sample_path)
                logger.info(f"Loaded voice sample from {self.voice_sample_path}")

            self._initialized = True
            logger.info("Chatterbox TTS model initialized successfully")

        except ImportError:
            logger.warning(
                "Chatterbox TTS not installed. TTS will return mock audio. "
                "Install with: pip install chatterbox-tts"
            )
            self._initialized = True  # Allow mock operation

        except Exception as e:
            logger.error(f"Failed to initialize TTS model: {e}")
            raise

    def _ensure_initialized(self) -> None:
        """초기화 확인"""
        if not self._initialized:
            raise RuntimeError("TTS module not initialized. Call initialize() first.")

    def _load_voice_sample(self, path: str) -> np.ndarray:
        """음성 샘플 로드"""
        try:
            import soundfile as sf

            audio, sr = sf.read(path)
            if sr != self.sample_rate:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            return audio
        except Exception as e:
            logger.warning(f"Failed to load voice sample: {e}")
            return None

    async def synthesize(
        self,
        text: str,
        voice_sample: Optional[Union[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        텍스트를 음성으로 변환 (동기식, 전체 반환)

        Args:
            text: 변환할 텍스트
            voice_sample: 음성 클로닝용 참조 오디오

        Returns:
            오디오 데이터 (numpy array)
        """
        self._ensure_initialized()

        if not text.strip():
            return np.array([], dtype=np.float32)

        # 음성 샘플 결정
        sample = voice_sample
        if sample is None:
            sample = self._voice_sample
        elif isinstance(sample, str):
            sample = self._load_voice_sample(sample)

        if self._model is None:
            # Mock audio for testing
            logger.debug("Using mock TTS audio (model not loaded)")
            return self._generate_mock_audio(len(text))

        try:
            # Chatterbox TTS 추론
            audio = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._model.generate(
                    text=text,
                    audio_prompt=sample,
                ),
            )

            return audio.cpu().numpy().flatten()

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return self._generate_mock_audio(len(text))

    async def synthesize_stream(
        self,
        text: str,
        chunk_size: int = 4096,
        voice_sample: Optional[Union[str, np.ndarray]] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        텍스트를 음성으로 변환 (스트리밍)

        Args:
            text: 변환할 텍스트
            chunk_size: 청크 크기 (샘플 수)
            voice_sample: 음성 클로닝용 참조 오디오

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

        # 문장 단위로 분리하여 스트리밍
        sentences = self._split_into_sentences(text)

        for i, sentence in enumerate(sentences):
            is_last = i == len(sentences) - 1

            # 각 문장 합성
            audio = await self.synthesize(sentence, voice_sample)

            if len(audio) == 0:
                continue

            # 청크로 분할하여 yield
            for j in range(0, len(audio), chunk_size):
                chunk_audio = audio[j : j + chunk_size]
                chunk_bytes = self._audio_to_bytes(chunk_audio)

                duration_ms = len(chunk_audio) / self.sample_rate * 1000
                is_chunk_last = is_last and (j + chunk_size >= len(audio))

                yield TTSChunk(
                    audio_data=chunk_bytes,
                    sample_rate=self.sample_rate,
                    duration_ms=duration_ms,
                    is_last=is_chunk_last,
                )

                # 실시간 스트리밍 시뮬레이션을 위한 작은 지연
                await asyncio.sleep(duration_ms / 1000 * 0.1)

    async def synthesize_stream_realtime(
        self,
        text_stream: AsyncGenerator[str, None],
        chunk_size: int = 4096,
        voice_sample: Optional[Union[str, np.ndarray]] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        LLM 스트리밍 출력을 실시간으로 TTS 변환

        Args:
            text_stream: 텍스트 청크 스트림 (LLM 출력)
            chunk_size: 오디오 청크 크기
            voice_sample: 음성 클로닝용 참조 오디오

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
                    sentence, chunk_size, voice_sample
                ):
                    # 마지막 여부는 전체 스트림에서 결정되므로 False로 설정
                    audio_chunk.is_last = False
                    yield audio_chunk

        # 남은 버퍼 처리
        if buffer.strip():
            async for audio_chunk in self.synthesize_stream(
                buffer, chunk_size, voice_sample
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
        # 텍스트 길이에 비례한 무음 오디오
        duration_samples = int(text_length * 0.1 * self.sample_rate)
        return np.zeros(duration_samples, dtype=np.float32)

    async def cleanup(self) -> None:
        """리소스 정리"""
        if self._model is not None:
            del self._model
            self._model = None
        self._voice_sample = None
        self._initialized = False
        logger.info("TTS module cleaned up")
