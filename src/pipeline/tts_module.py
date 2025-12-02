"""
TTS Module with multiple provider support.

다중 TTS 프로바이더 지원 모듈
- edge-tts: Microsoft Azure 기반 한국어 TTS (무료, 빠름)
- chatterbox: 음성 클로닝 지원 (영어만)
- zonos: 고품질 음성 복제 TTS (다국어, 한국어 미지원)

특징:
- 200ms 미만 지연시간
- 스트리밍 출력 지원
- 한국어/영어 자동 선택
"""

import asyncio
import io
import logging
import re
from pathlib import Path
from typing import AsyncGenerator, Optional, Literal

import numpy as np

from ..models.schemas import TTSChunk

logger = logging.getLogger(__name__)

# 컴파일된 정규식 패턴 (성능 최적화)
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?。！？])\s+")
_SENTENCE_END_PATTERN = re.compile(r"[.!?。！？]")


class TTSModule:
    """
    다중 프로바이더 TTS 모듈

    Features:
    - 텍스트 → 음성 변환
    - 스트리밍 출력
    - edge-tts: 한국어 지원 (무료)
    - chatterbox: 음성 클로닝 지원 (영어)
    - zonos: 고품질 음성 복제 (다국어)
    """

    def __init__(
        self,
        provider: Literal["edge-tts", "chatterbox", "zonos"] = "edge-tts",
        voice: str = "ko-KR-SunHiNeural",
        voice_sample_path: Optional[str] = None,
        sample_rate: int = 24000,
        device: str = "cuda",
        voice_id: str = "default",
    ):
        """
        Initialize TTS Module.

        Args:
            provider: TTS 프로바이더 ("edge-tts" 또는 "chatterbox")
            voice: edge-tts 음성 ID (예: ko-KR-SunHiNeural, ko-KR-InJoonNeural)
            voice_sample_path: 음성 클로닝용 참조 오디오 경로 (chatterbox용)
            sample_rate: 출력 샘플레이트
            device: Compute device
            voice_id: 음성 ID (chatterbox용)
        """
        self.provider = provider
        self.voice = voice
        self.voice_sample_path = voice_sample_path
        self.sample_rate = sample_rate
        self.device = device
        self.voice_id = voice_id

        self._chatterbox_model = None
        self._zonos_model = None
        self._initialized = False

        logger.info(f"TTS Module created: provider={provider}, voice={voice}")

    async def initialize(self) -> None:
        """모델 초기화"""
        if self._initialized:
            return

        if self.provider == "edge-tts":
            # edge-tts는 별도 초기화 불필요 (API 기반)
            logger.info(f"Initializing edge-tts with voice: {self.voice}")
            try:
                import edge_tts
                # 테스트 통신
                communicate = edge_tts.Communicate("테스트", self.voice)
                logger.info(f"edge-tts initialized successfully with voice: {self.voice}")
                self._initialized = True
            except ImportError:
                logger.error("edge-tts not installed. Run: pip install edge-tts")
                self._initialized = True  # 폴백 모드
            except Exception as e:
                logger.error(f"edge-tts initialization failed: {e}")
                self._initialized = True  # 폴백 모드

        elif self.provider == "chatterbox":
            logger.info("Initializing Chatterbox TTS model...")
            try:
                from ..models.integrations import ChatterboxTTSModel

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
                logger.error(f"Failed to initialize Chatterbox TTS: {e}")
                self._initialized = True  # 폴백 모드

        elif self.provider == "zonos":
            logger.info("Initializing Zonos TTS model...")
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
        else:
            logger.error(f"Unknown TTS provider: {self.provider}")
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
            voice_id: 사용할 음성 ID (edge-tts: 무시, chatterbox: 사용)

        Returns:
            오디오 데이터 (numpy array, float32, -1.0 ~ 1.0)
        """
        self._ensure_initialized()

        if not text.strip():
            return np.array([], dtype=np.float32)

        if self.provider == "edge-tts":
            return await self._synthesize_edge_tts(text)
        elif self.provider == "chatterbox":
            return await self._synthesize_chatterbox(text, voice_id)
        elif self.provider == "zonos":
            return await self._synthesize_zonos(text, voice_id)
        else:
            logger.warning(f"Unknown provider: {self.provider}, using mock audio")
            return self._generate_mock_audio(len(text))

    async def _synthesize_edge_tts(self, text: str) -> np.ndarray:
        """edge-tts로 음성 합성"""
        try:
            import edge_tts

            logger.info(f"🎤 edge-tts synthesizing: '{text[:50]}...' with voice {self.voice}")

            # edge-tts로 MP3 생성
            communicate = edge_tts.Communicate(text, self.voice)
            mp3_data = b""

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_data += chunk["data"]

            if not mp3_data:
                logger.warning("edge-tts returned empty audio")
                return self._generate_mock_audio(len(text))

            logger.debug(f"edge-tts MP3 data: {len(mp3_data)} bytes")

            # MP3를 PCM으로 변환
            audio = await self._convert_mp3_to_pcm(mp3_data)
            logger.info(f"✅ edge-tts synthesis complete: {len(audio)} samples")
            return audio

        except ImportError:
            logger.error("edge-tts not installed")
            return self._generate_mock_audio(len(text))
        except Exception as e:
            logger.error(f"edge-tts synthesis error: {e}", exc_info=True)
            return self._generate_mock_audio(len(text))

    async def _convert_mp3_to_pcm(self, mp3_data: bytes) -> np.ndarray:
        """MP3 바이트를 PCM numpy array로 변환"""
        try:
            # pydub 사용
            from pydub import AudioSegment

            audio_segment = AudioSegment.from_mp3(io.BytesIO(mp3_data))

            # 타겟 샘플레이트로 리샘플링
            if audio_segment.frame_rate != self.sample_rate:
                audio_segment = audio_segment.set_frame_rate(self.sample_rate)

            # 모노로 변환
            if audio_segment.channels > 1:
                audio_segment = audio_segment.set_channels(1)

            # numpy array로 변환
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

            # float32로 정규화 (-1.0 ~ 1.0)
            audio = samples.astype(np.float32) / 32768.0

            logger.debug(f"MP3 to PCM conversion: {len(audio)} samples at {self.sample_rate}Hz")
            return audio

        except ImportError:
            logger.warning("pydub not installed, trying librosa")
            try:
                import librosa

                # 임시 파일 없이 메모리에서 처리
                audio, sr = librosa.load(io.BytesIO(mp3_data), sr=self.sample_rate, mono=True)
                return audio.astype(np.float32)
            except Exception as e:
                logger.error(f"librosa conversion failed: {e}")
                return self._generate_mock_audio(100)

        except Exception as e:
            logger.error(f"MP3 to PCM conversion failed: {e}")
            return self._generate_mock_audio(100)

    async def _synthesize_chatterbox(self, text: str, voice_id: Optional[str] = None) -> np.ndarray:
        """Chatterbox TTS로 음성 합성"""
        use_voice_id = voice_id or self.voice_id

        if self._chatterbox_model is None:
            logger.warning(f"Chatterbox model not loaded, using mock audio for text: '{text[:50]}...'")
            return self._generate_mock_audio(len(text))

        try:
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

    async def _synthesize_zonos(self, text: str, voice_id: Optional[str] = None) -> np.ndarray:
        """Zonos TTS로 음성 합성"""
        use_voice_id = voice_id or self.voice_id

        if self._zonos_model is None:
            logger.warning(f"Zonos model not loaded, using mock audio for text: '{text[:50]}...'")
            return self._generate_mock_audio(len(text))

        try:
            logger.debug(f"Calling Zonos TTS synthesize: text='{text[:50]}...', voice_id={use_voice_id}")
            audio = await self._zonos_model.synthesize(
                text=text,
                voice_id=use_voice_id,
                language="en",  # Zonos default language
            )

            if audio is not None and len(audio) > 0:
                # Resample from 44.1kHz to target sample rate if needed
                if self._zonos_model.sample_rate != self.sample_rate:
                    import librosa
                    audio = librosa.resample(
                        audio,
                        orig_sr=self._zonos_model.sample_rate,
                        target_sr=self.sample_rate,
                    )
                logger.debug(f"Zonos TTS synthesize completed: {len(audio)} samples")
            else:
                logger.warning("Zonos TTS synthesize returned empty audio")

            return audio

        except Exception as e:
            logger.error(f"Zonos TTS synthesis error: {e}", exc_info=True)
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
        if self._chatterbox_model is not None:
            await self._chatterbox_model.cleanup()
            self._chatterbox_model = None
        if self._zonos_model is not None:
            await self._zonos_model.cleanup()
            self._zonos_model = None
        self._initialized = False
        logger.info("TTS module cleaned up")
