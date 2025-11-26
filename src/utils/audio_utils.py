"""
Audio processing utilities.

오디오 처리 유틸리티 모듈
"""

import io
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    오디오 처리 유틸리티

    Features:
    - 포맷 변환
    - 리샘플링
    - 정규화
    - 청크 처리
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        target_dtype: str = "float32",
    ):
        """
        Initialize Audio Processor.

        Args:
            target_sample_rate: 목표 샘플레이트
            target_channels: 목표 채널 수
            target_dtype: 목표 데이터 타입
        """
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.target_dtype = target_dtype

    def bytes_to_array(
        self,
        audio_bytes: bytes,
        dtype: str = "int16",
    ) -> np.ndarray:
        """
        bytes를 numpy array로 변환

        Args:
            audio_bytes: 오디오 바이트 데이터
            dtype: 입력 데이터 타입

        Returns:
            numpy array
        """
        dtype_map = {
            "int16": np.int16,
            "int32": np.int32,
            "float32": np.float32,
            "float64": np.float64,
        }

        np_dtype = dtype_map.get(dtype, np.int16)
        return np.frombuffer(audio_bytes, dtype=np_dtype)

    def array_to_bytes(
        self,
        audio_array: np.ndarray,
        dtype: str = "int16",
    ) -> bytes:
        """
        numpy array를 bytes로 변환

        Args:
            audio_array: 오디오 배열
            dtype: 출력 데이터 타입

        Returns:
            bytes
        """
        if dtype == "int16":
            if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                audio_array = (audio_array * 32767).clip(-32768, 32767)
            return audio_array.astype(np.int16).tobytes()
        elif dtype == "float32":
            return audio_array.astype(np.float32).tobytes()
        else:
            return audio_array.tobytes()

    def normalize(
        self,
        audio: np.ndarray,
        target_level: float = 0.9,
    ) -> np.ndarray:
        """
        오디오 정규화

        Args:
            audio: 입력 오디오
            target_level: 목표 피크 레벨 (0-1)

        Returns:
            정규화된 오디오
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio * (target_level / max_val)
        return audio

    def resample(
        self,
        audio: np.ndarray,
        orig_sr: int,
        target_sr: Optional[int] = None,
    ) -> np.ndarray:
        """
        오디오 리샘플링

        Args:
            audio: 입력 오디오
            orig_sr: 원본 샘플레이트
            target_sr: 목표 샘플레이트 (없으면 기본값 사용)

        Returns:
            리샘플링된 오디오
        """
        target_sr = target_sr or self.target_sample_rate

        if orig_sr == target_sr:
            return audio

        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # 간단한 선형 보간 리샘플링
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            return np.interp(
                np.linspace(0, len(audio), new_length),
                np.arange(len(audio)),
                audio,
            )

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        스테레오를 모노로 변환

        Args:
            audio: 입력 오디오

        Returns:
            모노 오디오
        """
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            return audio.mean(axis=1)
        return audio

    def convert_format(
        self,
        audio: np.ndarray,
        from_dtype: str = "int16",
        to_dtype: str = "float32",
    ) -> np.ndarray:
        """
        오디오 데이터 타입 변환

        Args:
            audio: 입력 오디오
            from_dtype: 입력 타입
            to_dtype: 출력 타입

        Returns:
            변환된 오디오
        """
        # int16 → float32
        if from_dtype == "int16" and to_dtype == "float32":
            return audio.astype(np.float32) / 32768.0

        # float32 → int16
        if from_dtype == "float32" and to_dtype == "int16":
            return (audio * 32767).clip(-32768, 32767).astype(np.int16)

        return audio.astype(getattr(np, to_dtype))

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """
        전체 전처리 파이프라인

        Args:
            audio: 입력 오디오
            sample_rate: 입력 샘플레이트

        Returns:
            처리된 오디오
        """
        # 모노 변환
        audio = self.to_mono(audio)

        # float32 변환
        if audio.dtype == np.int16:
            audio = self.convert_format(audio, "int16", "float32")

        # 리샘플링
        audio = self.resample(audio, sample_rate)

        # 정규화
        audio = self.normalize(audio)

        return audio

    def chunk_audio(
        self,
        audio: np.ndarray,
        chunk_size: int,
        overlap: int = 0,
    ) -> list:
        """
        오디오를 청크로 분할

        Args:
            audio: 입력 오디오
            chunk_size: 청크 크기 (샘플 수)
            overlap: 오버랩 샘플 수

        Returns:
            청크 리스트
        """
        chunks = []
        step = chunk_size - overlap

        for i in range(0, len(audio) - chunk_size + 1, step):
            chunks.append(audio[i:i + chunk_size])

        # 남은 부분
        if len(audio) % step != 0:
            last_chunk = audio[-chunk_size:]
            if len(last_chunk) == chunk_size:
                chunks.append(last_chunk)

        return chunks

    def merge_chunks(
        self,
        chunks: list,
        overlap: int = 0,
    ) -> np.ndarray:
        """
        청크를 병합

        Args:
            chunks: 청크 리스트
            overlap: 오버랩 샘플 수

        Returns:
            병합된 오디오
        """
        if not chunks:
            return np.array([], dtype=np.float32)

        if overlap == 0:
            return np.concatenate(chunks)

        # 크로스페이드로 병합
        result = chunks[0].copy()
        fade = np.linspace(0, 1, overlap)

        for chunk in chunks[1:]:
            # 크로스페이드
            result[-overlap:] = (
                result[-overlap:] * (1 - fade) + chunk[:overlap] * fade
            )
            result = np.concatenate([result, chunk[overlap:]])

        return result

    def calculate_rms(self, audio: np.ndarray) -> float:
        """RMS 계산"""
        return np.sqrt(np.mean(audio**2))

    def calculate_db(self, audio: np.ndarray) -> float:
        """dB 레벨 계산"""
        rms = self.calculate_rms(audio)
        if rms > 0:
            return 20 * np.log10(rms)
        return -100.0

    def apply_fade(
        self,
        audio: np.ndarray,
        fade_in_samples: int = 0,
        fade_out_samples: int = 0,
    ) -> np.ndarray:
        """
        페이드 인/아웃 적용

        Args:
            audio: 입력 오디오
            fade_in_samples: 페이드 인 길이
            fade_out_samples: 페이드 아웃 길이

        Returns:
            페이드 적용된 오디오
        """
        audio = audio.copy()

        if fade_in_samples > 0:
            fade_in = np.linspace(0, 1, fade_in_samples)
            audio[:fade_in_samples] *= fade_in

        if fade_out_samples > 0:
            fade_out = np.linspace(1, 0, fade_out_samples)
            audio[-fade_out_samples:] *= fade_out

        return audio
