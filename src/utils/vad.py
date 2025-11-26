"""
Voice Activity Detection using Silero VAD.

Silero VAD 기반 음성 활동 감지 모듈
특징:
- 고정밀 음성 감지
- 저지연 처리
- MIT 라이선스 (상업적 사용 가능)
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """
    Silero VAD 기반 음성 활동 감지기

    Features:
    - 실시간 음성 감지
    - 음성 구간 추출
    - 무음 구간 감지
    """

    def __init__(
        self,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 300,
        window_size_samples: int = 512,
    ):
        """
        Initialize Voice Activity Detector.

        Args:
            threshold: VAD 임계값 (0-1)
            sample_rate: 샘플레이트
            min_speech_duration_ms: 최소 음성 길이 (ms)
            min_silence_duration_ms: 최소 무음 길이 (ms)
            window_size_samples: 윈도우 크기 (샘플)
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples

        self._model = None
        self._utils = None
        self._initialized = False

        # 상태
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_samples = 0

    def initialize(self) -> bool:
        """
        Silero VAD 모델 초기화

        Returns:
            초기화 성공 여부
        """
        if self._initialized:
            return True

        try:
            import torch

            # Silero VAD 모델 로드
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )

            self._model = model
            self._utils = utils
            self._initialized = True
            logger.info("Silero VAD initialized successfully")
            return True

        except ImportError:
            logger.warning(
                "PyTorch not installed. VAD will use simple energy-based detection."
            )
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            return False

    def is_speech(
        self,
        audio: np.ndarray,
        return_probability: bool = False,
    ) -> bool | Tuple[bool, float]:
        """
        오디오가 음성인지 감지

        Args:
            audio: 오디오 배열 (float32, 16kHz)
            return_probability: 확률 반환 여부

        Returns:
            음성 여부 (또는 (음성 여부, 확률))
        """
        if not self._initialized:
            self.initialize()

        if self._model is not None:
            prob = self._silero_detect(audio)
        else:
            prob = self._energy_detect(audio)

        is_speech = prob > self.threshold

        if return_probability:
            return is_speech, prob
        return is_speech

    def _silero_detect(self, audio: np.ndarray) -> float:
        """Silero VAD로 감지"""
        import torch

        # numpy → torch tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio

        # 배치 차원 추가
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        # VAD 추론
        with torch.no_grad():
            speech_prob = self._model(
                audio_tensor,
                self.sample_rate
            ).item()

        return speech_prob

    def _energy_detect(self, audio: np.ndarray) -> float:
        """
        에너지 기반 간단한 VAD (폴백)
        """
        # RMS 에너지 계산
        rms = np.sqrt(np.mean(audio**2))

        # dB로 변환 및 정규화
        if rms > 0:
            db = 20 * np.log10(rms)
            # -60dB ~ 0dB 범위를 0~1로 정규화
            prob = (db + 60) / 60
            prob = max(0, min(1, prob))
        else:
            prob = 0

        return prob

    def get_speech_segments(
        self,
        audio: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        음성 구간 추출

        Args:
            audio: 전체 오디오 배열

        Returns:
            음성 구간 리스트 [(start_sample, end_sample), ...]
        """
        if not self._initialized:
            self.initialize()

        segments = []
        speech_start = None

        min_speech_samples = int(
            self.min_speech_duration_ms * self.sample_rate / 1000
        )
        min_silence_samples = int(
            self.min_silence_duration_ms * self.sample_rate / 1000
        )

        # 윈도우 단위로 처리
        for i in range(0, len(audio) - self.window_size_samples, self.window_size_samples):
            window = audio[i:i + self.window_size_samples]
            is_speech = self.is_speech(window)

            if is_speech:
                if speech_start is None:
                    speech_start = i
                self._silence_samples = 0
            else:
                self._silence_samples += self.window_size_samples

                if speech_start is not None:
                    if self._silence_samples >= min_silence_samples:
                        # 음성 구간 종료
                        speech_end = i - self._silence_samples + self.window_size_samples
                        duration = speech_end - speech_start

                        if duration >= min_speech_samples:
                            segments.append((speech_start, speech_end))

                        speech_start = None
                        self._silence_samples = 0

        # 마지막 구간 처리
        if speech_start is not None:
            speech_end = len(audio)
            duration = speech_end - speech_start

            if duration >= min_speech_samples:
                segments.append((speech_start, speech_end))

        return segments

    def extract_speech(
        self,
        audio: np.ndarray,
        padding_ms: int = 50,
    ) -> List[np.ndarray]:
        """
        음성 구간만 추출

        Args:
            audio: 전체 오디오 배열
            padding_ms: 구간 전후 패딩 (ms)

        Returns:
            음성 구간 오디오 리스트
        """
        segments = self.get_speech_segments(audio)
        padding_samples = int(padding_ms * self.sample_rate / 1000)

        speech_clips = []
        for start, end in segments:
            # 패딩 추가
            start = max(0, start - padding_samples)
            end = min(len(audio), end + padding_samples)
            speech_clips.append(audio[start:end])

        return speech_clips

    def process_stream(
        self,
        audio_chunk: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        스트리밍 음성 처리

        Args:
            audio_chunk: 오디오 청크

        Returns:
            완성된 음성 구간 (없으면 None)
        """
        if not self._initialized:
            self.initialize()

        min_silence_samples = int(
            self.min_silence_duration_ms * self.sample_rate / 1000
        )
        min_speech_samples = int(
            self.min_speech_duration_ms * self.sample_rate / 1000
        )

        is_speech = self.is_speech(audio_chunk)

        if is_speech:
            self._speech_buffer.extend(audio_chunk.tolist())
            self._is_speaking = True
            self._silence_samples = 0
        else:
            if self._is_speaking:
                self._speech_buffer.extend(audio_chunk.tolist())
                self._silence_samples += len(audio_chunk)

                # 충분한 무음이 감지되면 발화 종료
                if self._silence_samples >= min_silence_samples:
                    if len(self._speech_buffer) >= min_speech_samples:
                        result = np.array(self._speech_buffer, dtype=np.float32)
                    else:
                        result = None

                    # 버퍼 리셋
                    self._speech_buffer = []
                    self._is_speaking = False
                    self._silence_samples = 0

                    return result

        return None

    def reset(self):
        """상태 리셋"""
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_samples = 0

    def get_state(self) -> dict:
        """현재 상태 반환"""
        return {
            "is_speaking": self._is_speaking,
            "buffer_length_ms": len(self._speech_buffer) * 1000 / self.sample_rate,
            "silence_duration_ms": self._silence_samples * 1000 / self.sample_rate,
        }
