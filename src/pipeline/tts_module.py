"""
TTS Module with Edge TTS, Zonos, and ElevenLabs TTS support.

TTS 음성 합성 모듈
- edge: Microsoft Edge TTS (무료, 빠름, 클라우드) ⭐ 권장
- zonos: 고품질 음성 복제 TTS (다국어, 한국어 포함, 로컬 실행, 느림)
- elevenlabs: ElevenLabs API 기반 고품질 TTS (다국어, 한국어 포함, 클라우드, 유료)

특징:
- Edge TTS: ~1초 응답 (가장 빠름!)
- 스트리밍 출력 지원
- 한국어/영어 자동 선택
"""

import asyncio
import logging
import re
from typing import AsyncGenerator, Optional, Literal, Union

import numpy as np

# 리샘플링용 librosa (모듈 레벨 import로 성능 최적화)
try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False
    librosa = None

from ..models.schemas import TTSChunk

logger = logging.getLogger(__name__)

# 컴파일된 정규식 패턴 (성능 최적화)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")
_SENTENCE_END_PATTERN = re.compile(r"[.!?。！？]")


class TTSModule:
    """
    TTS 모듈 (Edge TTS, Zonos, ElevenLabs 지원)

    Features:
    - 텍스트 → 음성 변환
    - 스트리밍 출력
    - edge: Microsoft Edge TTS (무료, 빠름) ⭐ 권장
    - zonos: 고품질 음성 복제 (다국어, 한국어 포함, 로컬 실행, 느림)
    - elevenlabs: ElevenLabs API 기반 고품질 TTS (다국어, 한국어 포함, 클라우드)
    """

    def __init__(
        self,
        provider: Literal["edge", "zonos", "elevenlabs"] = "edge",
        voice: str = "default",
        voice_sample_path: Optional[str] = None,
        sample_rate: int = 24000,
        device: str = "cuda",
        voice_id: str = "default",
        enable_compile: bool = False,
        # Edge TTS 설정
        edge_voice: Optional[str] = None,
        # ElevenLabs 설정
        elevenlabs_api_key: Optional[str] = None,
        elevenlabs_voice_id: Optional[str] = None,
        elevenlabs_model_id: Optional[str] = None,
    ):
        """
        Initialize TTS Module.

        Args:
            provider: TTS 프로바이더 ("edge", "zonos", "elevenlabs")
            voice: 음성 ID
            voice_sample_path: 음성 클로닝용 참조 오디오 경로 (Zonos용)
            sample_rate: 출력 샘플레이트
            device: Compute device (Zonos용)
            voice_id: 음성 ID
            enable_compile: torch.compile() 활성화 (Linux only, Zonos용)
            edge_voice: Edge TTS 음성 (예: "ko-KR-SunHiNeural")
            elevenlabs_api_key: ElevenLabs API 키
            elevenlabs_voice_id: ElevenLabs 음성 ID
            elevenlabs_model_id: ElevenLabs 모델 ID
        """
        self.provider = provider
        self.voice = voice
        self.voice_sample_path = voice_sample_path
        self.sample_rate = sample_rate
        self.device = device
        self.voice_id = voice_id
        self.enable_compile = enable_compile

        # Edge TTS 설정
        self.edge_voice = edge_voice or "ko-KR-SunHiNeural"

        # ElevenLabs 설정
        self.elevenlabs_api_key = elevenlabs_api_key
        self.elevenlabs_voice_id = elevenlabs_voice_id
        self.elevenlabs_model_id = elevenlabs_model_id

        # torch.compile() 설정 (ZonosTTS import 전에 환경변수 설정 필요)
        import os
        if enable_compile and provider == "zonos":
            os.environ["ZONOS_ENABLE_COMPILE"] = "1"
            logger.info("🚀 torch.compile() 활성화됨 (TTS 속도 향상)")

        self._edge_model = None
        self._zonos_model = None
        self._elevenlabs_model = None
        self._initialized = False

        logger.info(f"TTS Module created: provider={provider}, voice={voice}")

    async def initialize(self) -> None:
        """모델 초기화"""
        if self._initialized:
            return

        if self.provider == "edge":
            logger.info("Initializing Edge TTS model... (빠른 TTS)")
            try:
                from ..models.integrations.edge_tts import EdgeTTSModel

                self._edge_model = EdgeTTSModel(
                    default_voice=self.edge_voice,
                    sample_rate=self.sample_rate,
                )

                success = await self._edge_model.initialize()

                if not success:
                    logger.warning("Edge TTS initialization returned False, using fallback")

                self._initialized = True
                logger.info("✅ Edge TTS model initialized successfully (빠른 응답!)")

            except Exception as e:
                logger.error(f"Failed to initialize Edge TTS: {e}")
                self._initialized = True  # 폴백 모드

        elif self.provider == "zonos":
            logger.info("Initializing Zonos TTS model... (느린 TTS)")
            try:
                from ..models.integrations import ZonosTTSModel

                self._zonos_model = ZonosTTSModel(
                    device=self.device,
                    sample_rate=44100,  # Zonos uses 44.1kHz
                )

                success = await self._zonos_model.initialize()

                if not success:
                    logger.warning("Zonos TTS initialization returned False, using fallback")

                self._initialized = True
                logger.info("Zonos TTS model initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize Zonos TTS: {e}")
                self._initialized = True  # 폴백 모드

        elif self.provider == "elevenlabs":
            logger.info("Initializing ElevenLabs TTS model...")
            try:
                from ..models.integrations import ElevenLabsTTSModel

                if not self.elevenlabs_api_key:
                    raise ValueError("ElevenLabs API key is required")

                self._elevenlabs_model = ElevenLabsTTSModel(
                    api_key=self.elevenlabs_api_key,
                    voice_id=self.elevenlabs_voice_id or self.voice_id,
                    model_id=self.elevenlabs_model_id or "eleven_multilingual_v2",
                    sample_rate=self.sample_rate,
                )

                success = await self._elevenlabs_model.initialize()

                if not success:
                    logger.warning("ElevenLabs TTS initialization returned False, using fallback")

                self._initialized = True
                logger.info("ElevenLabs TTS model initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize ElevenLabs TTS: {e}")
                self._initialized = True  # 폴백 모드

        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")

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
            오디오 데이터 (numpy array, float32, -1.0 ~ 1.0)
        """
        self._ensure_initialized()

        if not text.strip():
            return np.array([], dtype=np.float32)

        if self.provider == "edge":
            return await self._synthesize_edge(text, voice_id)
        elif self.provider == "zonos":
            return await self._synthesize_zonos(text, voice_id)
        elif self.provider == "elevenlabs":
            return await self._synthesize_elevenlabs(text, voice_id)
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")

    async def _synthesize_edge(self, text: str, voice_id: Optional[str] = None) -> np.ndarray:
        """Edge TTS로 음성 합성 (빠름!)"""
        if self._edge_model is None:
            logger.warning(f"Edge model not loaded, using mock audio for text: '{text[:50]}...'")
            return self._generate_mock_audio(len(text))

        try:
            logger.debug(f"Calling Edge TTS synthesize: text='{text[:50]}...'")
            audio = await self._edge_model.synthesize(
                text=text,
                language="ko",  # 한국어 기본값
            )

            if audio is not None and len(audio) > 0:
                logger.debug(f"Edge TTS synthesize completed: {len(audio)} samples")
            else:
                logger.warning("Edge TTS synthesize returned empty audio")

            return audio

        except Exception as e:
            logger.error(f"Edge TTS synthesis error: {e}", exc_info=True)
            logger.warning(f"Falling back to mock audio for text: '{text[:50]}...'")
            return self._generate_mock_audio(len(text))

    async def _synthesize_zonos(self, text: str, voice_id: Optional[str] = None) -> np.ndarray:
        """Zonos TTS로 음성 합성 (느림)"""
        use_voice_id = voice_id or self.voice_id

        if self._zonos_model is None:
            logger.warning(f"Zonos model not loaded, using mock audio for text: '{text[:50]}...'")
            return self._generate_mock_audio(len(text))

        try:
            logger.debug(f"Calling Zonos TTS synthesize: text='{text[:50]}...', voice_id={use_voice_id}")
            audio = await self._zonos_model.synthesize(
                text=text,
                voice_id=use_voice_id,
                language="ko",  # 한국어 기본값
            )

            if audio is not None and len(audio) > 0:
                # Resample from 44.1kHz to target sample rate if needed (모듈 레벨 librosa 사용)
                if self._zonos_model.sample_rate != self.sample_rate:
                    if _LIBROSA_AVAILABLE:
                        audio = librosa.resample(
                            audio,
                            orig_sr=self._zonos_model.sample_rate,
                            target_sr=self.sample_rate,
                        )
                    else:
                        logger.warning("librosa not available, skipping resampling")
                logger.debug(f"Zonos TTS synthesize completed: {len(audio)} samples")
            else:
                logger.warning("Zonos TTS synthesize returned empty audio")

            return audio

        except Exception as e:
            logger.error(f"Zonos TTS synthesis error: {e}", exc_info=True)
            logger.warning(f"Falling back to mock audio for text: '{text[:50]}...'")
            return self._generate_mock_audio(len(text))

    async def _synthesize_elevenlabs(self, text: str, voice_id: Optional[str] = None) -> np.ndarray:
        """ElevenLabs TTS로 음성 합성"""
        use_voice_id = voice_id or self.voice_id

        if self._elevenlabs_model is None:
            logger.warning(f"ElevenLabs model not loaded, using mock audio for text: '{text[:50]}...'")
            return self._generate_mock_audio(len(text))

        try:
            logger.debug(f"Calling ElevenLabs TTS synthesize: text='{text[:50]}...', voice_id={use_voice_id}")
            audio = await self._elevenlabs_model.synthesize(
                text=text,
                voice_id=use_voice_id,
                language="ko",  # 한국어 기본값
            )

            if audio is not None and len(audio) > 0:
                logger.debug(f"ElevenLabs TTS synthesize completed: {len(audio)} samples")
            else:
                logger.warning("ElevenLabs TTS synthesize returned empty audio")

            return audio

        except Exception as e:
            logger.error(f"ElevenLabs TTS synthesis error: {e}", exc_info=True)
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
        initial_chunk_size: int = 1024,
        min_text_for_early_synthesis: int = 20,
        voice_id: Optional[str] = None,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        LLM 스트리밍 출력을 실시간으로 TTS 변환 (지연시간 최적화)

        Args:
            text_stream: 텍스트 청크 스트림 (LLM 출력)
            chunk_size: 오디오 청크 크기
            initial_chunk_size: 첫 출력용 작은 청크 크기
            min_text_for_early_synthesis: 조기 합성 트리거 최소 텍스트 길이
            voice_id: 사용할 음성 ID

        Yields:
            TTSChunk: 오디오 청크
        """
        buffer = ""
        is_first_chunk = True
        total_synthesized = 0

        async for text_chunk in text_stream:
            buffer += text_chunk

            # 문장이 완성되면 즉시 합성
            sentences, remaining = self._extract_complete_sentences(buffer)

            # 문장이 완성되지 않았지만 충분히 길면 조기 합성 (첫 출력 지연 감소)
            if not sentences and len(buffer) >= min_text_for_early_synthesis:
                # 쉼표나 공백으로 끝나는 자연스러운 구간에서 분리
                split_point = self._find_early_split_point(buffer)
                if split_point > 0:
                    sentences = [buffer[:split_point]]
                    remaining = buffer[split_point:]

            buffer = remaining

            for sentence in sentences:
                # 첫 청크는 작은 크기로 빠른 출력
                current_chunk_size = initial_chunk_size if is_first_chunk else chunk_size

                async for audio_chunk in self.synthesize_stream(
                    sentence, current_chunk_size, voice_id
                ):
                    # 마지막 여부는 전체 스트림에서 결정되므로 False로 설정
                    audio_chunk.is_last = False
                    yield audio_chunk
                    total_synthesized += 1

                    # 첫 청크 출력 후 일반 크기로 전환
                    if is_first_chunk:
                        is_first_chunk = False
                        logger.debug(f"First TTS chunk sent, switching to normal chunk size")

        # 남은 버퍼 처리
        if buffer.strip():
            async for audio_chunk in self.synthesize_stream(
                buffer, chunk_size, voice_id
            ):
                yield audio_chunk
                total_synthesized += 1

        logger.debug(f"Realtime TTS stream completed: {total_synthesized} chunks sent")

    def _find_early_split_point(self, text: str) -> int:
        """
        조기 합성을 위한 자연스러운 분리 지점 찾기

        Args:
            text: 분석할 텍스트

        Returns:
            분리 지점 인덱스 (없으면 0)
        """
        # 우선순위: 쉼표 > 공백 (단어 경계)
        # 최소 10자 이상에서만 분리
        if len(text) < 10:
            return 0

        # 쉼표 위치 찾기
        comma_pos = text.rfind(',', 10, len(text) - 5)
        if comma_pos > 0:
            return comma_pos + 1

        # 공백 위치 찾기 (단어 경계)
        space_pos = text.rfind(' ', 10, len(text) - 5)
        if space_pos > 0:
            return space_pos + 1

        return 0

    async def synthesize_sentences_streaming(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "ko",
    ) -> AsyncGenerator[tuple, None]:
        """
        실시간 문장 단위 TTS 스트리밍

        문장별로 TTS를 생성하여 즉시 반환합니다.
        첫 문장 생성이 완료되면 바로 재생을 시작할 수 있어
        첫 응답 지연시간을 크게 줄일 수 있습니다.

        Args:
            text: 변환할 전체 텍스트
            voice_id: 사용할 음성 프로필 ID
            language: 언어 코드

        Yields:
            (audio_array, sentence_text, sentence_index, total_sentences) 튜플
            - audio_array: 해당 문장의 오디오 데이터 (numpy array)
            - sentence_text: 원본 문장 텍스트
            - sentence_index: 현재 문장 인덱스 (0부터)
            - total_sentences: 전체 문장 수
        """
        self._ensure_initialized()

        if not text.strip():
            return

        use_voice_id = voice_id or self.voice_id

        if self.provider == "edge":
            if self._edge_model is None:
                logger.warning("Edge model not loaded, using mock streaming")
                audio = self._generate_mock_audio(len(text))
                yield (audio, text, 0, 1)
                return

            try:
                # EdgeTTSModel의 문장 스트리밍 메서드 호출
                async for audio, sentence, idx, total in self._edge_model.synthesize_sentences_streaming(
                    text=text,
                    language=language,
                ):
                    yield (audio, sentence, idx, total)

            except Exception as e:
                logger.error(f"Edge sentence streaming error: {e}", exc_info=True)
                # 폴백: 전체 텍스트를 단일 문장으로 처리
                audio = await self.synthesize(text, voice_id)
                yield (audio, text, 0, 1)

        elif self.provider == "zonos":
            if self._zonos_model is None:
                logger.warning("Zonos model not loaded, using mock streaming")
                # 폴백: 단일 문장으로 처리
                audio = self._generate_mock_audio(len(text))
                yield (audio, text, 0, 1)
                return

            try:
                # ZonosTTSModel의 문장 스트리밍 메서드 호출
                async for audio, sentence, idx, total in self._zonos_model.synthesize_sentences_streaming(
                    text=text,
                    voice_id=use_voice_id,
                    language=language,
                ):
                    # 리샘플링 필요 시 적용 (모듈 레벨 librosa 사용)
                    if audio is not None and len(audio) > 0:
                        if self._zonos_model.sample_rate != self.sample_rate:
                            if _LIBROSA_AVAILABLE:
                                audio = librosa.resample(
                                    audio,
                                    orig_sr=self._zonos_model.sample_rate,
                                    target_sr=self.sample_rate,
                                )
                            else:
                                logger.warning("librosa not available, skipping resampling")

                    yield (audio, sentence, idx, total)

            except Exception as e:
                logger.error(f"Sentence streaming error: {e}", exc_info=True)
                # 폴백: 전체 텍스트를 단일 문장으로 처리
                audio = await self.synthesize(text, voice_id)
                yield (audio, text, 0, 1)

        elif self.provider == "elevenlabs":
            if self._elevenlabs_model is None:
                logger.warning("ElevenLabs model not loaded, using mock streaming")
                # 폴백: 단일 문장으로 처리
                audio = self._generate_mock_audio(len(text))
                yield (audio, text, 0, 1)
                return

            try:
                # ElevenLabsTTSModel의 문장 스트리밍 메서드 호출
                async for audio, sentence, idx, total in self._elevenlabs_model.synthesize_sentences_streaming(
                    text=text,
                    voice_id=use_voice_id,
                    language=language,
                ):
                    yield (audio, sentence, idx, total)

            except Exception as e:
                logger.error(f"Sentence streaming error: {e}", exc_info=True)
                # 폴백: 전체 텍스트를 단일 문장으로 처리
                audio = await self.synthesize(text, voice_id)
                yield (audio, text, 0, 1)
        else:
            raise ValueError(f"Unknown TTS provider: {self.provider}")

    def _split_into_sentences(self, text: str) -> list:
        """텍스트를 문장 단위로 분리"""
        sentences = _SENTENCE_SPLIT_PATTERN.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_complete_sentences(self, text: str) -> tuple:
        """
        완성된 문장 추출

        Returns:
            (완성된 문장 리스트, 남은 텍스트)
        """
        # 문장 종결 부호 위치 찾기
        matches = list(_SENTENCE_END_PATTERN.finditer(text))

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
        if self._edge_model is not None:
            await self._edge_model.cleanup()
            self._edge_model = None
        if self._zonos_model is not None:
            await self._zonos_model.cleanup()
            self._zonos_model = None
        if self._elevenlabs_model is not None:
            await self._elevenlabs_model.cleanup()
            self._elevenlabs_model = None
        self._initialized = False
        logger.info("TTS module cleaned up")
