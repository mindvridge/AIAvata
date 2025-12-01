"""
STT Module using SenseVoice-Small.

SenseVoice-Small 기반 음성 인식 + 감정 인식 모듈
특징:
- 한국어/영어/중국어/일본어 지원
- 감정인식 내장 (happy, sad, angry, neutral)
- Whisper 대비 5배 빠른 추론
- Apache 2.0 라이선스 (상업적 사용 가능)
"""

import asyncio
import time
from typing import Optional
import logging

import numpy as np

from ..models.emotion import Emotion, EmotionMapping
from ..models.schemas import STTResult

logger = logging.getLogger(__name__)


class STTModule:
    """
    SenseVoice-Small 기반 STT + 감정인식 모듈

    Features:
    - 실시간 음성 인식
    - 감정 인식 (happy, sad, angry, neutral)
    - 다국어 지원 (ko, en, zh, ja)
    - VAD 통합
    """

    def __init__(
        self,
        device: str = "cuda",
        vad_enabled: bool = True,
        max_segment_time: int = 30000,
        model_path: str = "iic/SenseVoiceSmall",
    ):
        """
        Initialize STT Module.

        Args:
            device: Compute device ('cuda', 'cpu', 'mps')
            vad_enabled: Whether to enable VAD
            max_segment_time: Maximum segment time in milliseconds
            model_path: ModelScope repository path for STT model
        """
        self.device = device
        self.vad_enabled = vad_enabled
        self.max_segment_time = max_segment_time
        self.model_path = model_path
        self.model = None
        self._initialized = False

    async def initialize(self) -> None:
        """모델 초기화 (비동기)"""
        if self._initialized:
            return

        logger.info("Initializing SenseVoice-Small model...")

        try:
            # FunASR 라이브러리에서 AutoModel 임포트
            from funasr import AutoModel

            # SenseVoice-Small 모델 로드
            # 여러 경로를 시도 (일반적으로 사용되는 경로들)
            model_paths = [
                self.model_path,  # 설정된 경로
                "iic/SenseVoiceSmall",  # ModelScope 일반 경로
                "iic/sensevoice_small",
                "FunAudioLLM/SenseVoiceSmall",  # 원래 경로
            ]
            
            model_loaded = False
            last_error = None
            
            for model_path in model_paths:
                try:
                    logger.info(f"Trying to load STT model from: {model_path}")
                    self.model = AutoModel(
                        model=model_path,
                        vad_model="fsmn-vad" if self.vad_enabled else None,
                        vad_kwargs={"max_single_segment_time": self.max_segment_time}
                        if self.vad_enabled
                        else None,
                        device=self.device,
                        disable_update=True,
                    )
                    model_loaded = True
                    logger.info(f"Successfully loaded STT model from: {model_path}")
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"Failed to load model from {model_path}: {e}")
                    continue
            
            if not model_loaded:
                raise RuntimeError(
                    f"Failed to load STT model from any of the attempted paths: {model_paths}. "
                    f"Last error: {last_error}"
                )

            self._initialized = True
            logger.info("SenseVoice-Small model initialized successfully")

        except ImportError:
            logger.warning(
                "FunASR not installed. STT will return mock results. "
                "Install with: pip install funasr"
            )
            self._initialized = True  # Allow operation with mock results

        except Exception as e:
            logger.error(f"Failed to initialize SenseVoice model: {e}")
            # STT 모델 초기화 실패 시에도 서비스는 계속 실행 (mock 모드)
            logger.warning("STT will operate in mock mode. Speech recognition will return placeholder text.")
            self._initialized = True  # Allow operation with mock results

    def _ensure_initialized(self) -> None:
        """모델이 초기화되었는지 확인"""
        if not self._initialized:
            raise RuntimeError("STT module not initialized. Call initialize() first.")

    async def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = "auto",
    ) -> STTResult:
        """
        오디오를 텍스트로 변환하고 감정을 분석합니다.

        Args:
            audio: numpy array 형태의 오디오 데이터 (float32, mono)
            sample_rate: 샘플레이트 (기본 16000)
            language: 언어 코드 ('auto', 'ko', 'en', 'zh', 'ja')

        Returns:
            STTResult: 인식 결과 (텍스트, 감정, 언어, 신뢰도)
        """
        self._ensure_initialized()
        start_time = time.time()

        # 오디오 전처리
        audio = self._preprocess_audio(audio, sample_rate)

        if self.model is None:
            # Mock result for testing without model
            logger.warning("STT model not loaded - returning mock result. Check model initialization.")
            return STTResult(
                text="",  # 빈 텍스트 반환 (Mock 데이터가 히스토리에 저장되지 않도록)
                emotion=Emotion.NEUTRAL,
                language="unknown",
                confidence=0.0,
                is_final=True,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        try:
            # SenseVoice 추론 실행
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate(
                    input=audio,
                    cache={},
                    language=language,
                    use_itn=True,  # Inverse text normalization
                    batch_size_s=0,  # Streaming mode
                ),
            )

            processing_time = (time.time() - start_time) * 1000

            # 결과 파싱
            if result and len(result) > 0:
                raw_result = result[0]
                text = raw_result.get("text", "")
                emotion_id = raw_result.get("emotion", 2)  # default: neutral
                detected_language = raw_result.get("language", "unknown")
                confidence = raw_result.get("confidence", 0.0)

                # SenseVoice 감정 ID를 Emotion enum으로 변환
                emotion = EmotionMapping.from_sensevoice(emotion_id)

                return STTResult(
                    text=text.strip(),
                    emotion=emotion,
                    language=detected_language,
                    confidence=confidence,
                    is_final=True,
                    processing_time_ms=processing_time,
                )

            return STTResult(
                text="",
                emotion=Emotion.NEUTRAL,
                language="unknown",
                confidence=0.0,
                is_final=True,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            return STTResult(
                text="",
                emotion=Emotion.NEUTRAL,
                language="unknown",
                confidence=0.0,
                is_final=True,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def transcribe_stream(
        self,
        audio_stream,
        sample_rate: int = 16000,
        language: str = "auto",
    ):
        """
        스트리밍 오디오 처리

        Args:
            audio_stream: AsyncGenerator yielding audio chunks
            sample_rate: 샘플레이트
            language: 언어 코드

        Yields:
            STTResult: 부분/최종 인식 결과
        """
        self._ensure_initialized()

        buffer = []
        min_chunk_duration = 0.5  # 최소 500ms 버퍼링
        samples_per_chunk = int(sample_rate * min_chunk_duration)

        async for chunk in audio_stream:
            buffer.extend(chunk)

            if len(buffer) >= samples_per_chunk:
                audio_array = np.array(buffer, dtype=np.float32)
                result = await self.transcribe(audio_array, sample_rate, language)

                if result.text:
                    yield result

                buffer = []

        # 남은 버퍼 처리
        if buffer:
            audio_array = np.array(buffer, dtype=np.float32)
            result = await self.transcribe(audio_array, sample_rate, language)
            result.is_final = True
            if result.text:
                yield result

    def _preprocess_audio(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        오디오 전처리

        Args:
            audio: 입력 오디오 배열
            sample_rate: 현재 샘플레이트

        Returns:
            전처리된 오디오 (16kHz, mono, float32)
        """
        # float32로 변환
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
            else:
                audio = audio.astype(np.float32)

        # 스테레오 → 모노
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = audio.mean(axis=1)

        # 리샘플링 (16kHz가 아닌 경우)
        if sample_rate != 16000:
            try:
                import librosa

                audio = librosa.resample(
                    audio, orig_sr=sample_rate, target_sr=16000
                )
            except ImportError:
                # 간단한 리샘플링 (librosa 없을 경우)
                ratio = 16000 / sample_rate
                new_length = int(len(audio) * ratio)
                audio = np.interp(
                    np.linspace(0, len(audio), new_length),
                    np.arange(len(audio)),
                    audio,
                )

        # 정규화
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        return audio

    async def cleanup(self) -> None:
        """리소스 정리"""
        if self.model is not None:
            del self.model
            self.model = None
        self._initialized = False
        logger.info("STT module cleaned up")
