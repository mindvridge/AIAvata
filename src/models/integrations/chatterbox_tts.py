"""
Chatterbox TTS Integration Module.

고품질 텍스트-음성 변환
GitHub: https://github.com/resemble-ai/chatterbox
라이선스: MIT (상업적 사용 가능)

특징:
- 200ms 미만 지연시간
- 음성 클로닝
- 다국어 지원
- 스트리밍 출력
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, AsyncGenerator, List

import numpy as np

logger = logging.getLogger(__name__)

# Chatterbox 모델 경로
CHATTERBOX_MODEL_DIR = os.getenv("CHATTERBOX_MODEL_DIR", "models/chatterbox")


class ChatterboxTTSModel:
    """
    Chatterbox TTS 모델

    텍스트를 자연스러운 음성으로 변환합니다.
    """

    def __init__(
        self,
        model_dir: str = CHATTERBOX_MODEL_DIR,
        device: str = "cuda",
        sample_rate: int = 24000,
    ):
        """
        Initialize Chatterbox TTS Model.

        Args:
            model_dir: 모델 파일 디렉토리
            device: 연산 디바이스
            sample_rate: 출력 샘플레이트
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.sample_rate = sample_rate

        # 모델
        self._model = None
        self._voice_encoder = None

        # 음성 임베딩 캐시
        self._voice_embeddings: dict = {}

        self._initialized = False

    async def initialize(self) -> bool:
        """
        모델 초기화 및 로드

        Returns:
            초기화 성공 여부
        """
        if self._initialized:
            return True

        logger.info("Initializing Chatterbox TTS model...")

        try:
            # Chatterbox 패키지 임포트 시도
            try:
                from chatterbox.tts import ChatterboxTTS

                # 사전 훈련된 모델 로드
                self._model = ChatterboxTTS.from_pretrained(device=self.device)
                logger.info("Chatterbox TTS model loaded from pretrained")

            except ImportError:
                logger.warning(
                    "Chatterbox TTS not installed. Trying alternative approach..."
                )

                # 대안: 로컬 모델 로드 시도
                if not self.model_dir.exists():
                    logger.warning(f"Model directory not found: {self.model_dir}")
                    self.model_dir.mkdir(parents=True, exist_ok=True)

                # 폴백 TTS 사용
                self._model = self._create_fallback_tts()

            except Exception as e:
                logger.warning(f"Failed to load Chatterbox: {e}. Using fallback.")
                self._model = self._create_fallback_tts()

            self._initialized = True
            logger.info("Chatterbox TTS initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Chatterbox TTS: {e}")
            return False

    def _create_fallback_tts(self):
        """폴백 TTS 생성 (EdgeTTS → gTTS → pyttsx3)"""
        # Edge TTS 시도 (고품질, 무료) - 네트워크 연결 테스트 포함
        try:
            import edge_tts
            import asyncio

            class EdgeTTSFallback:
                def __init__(self, sample_rate):
                    self.sample_rate = sample_rate
                    self.voice = "ko-KR-SunHiNeural"  # 한국어 여성 음성
                    self._network_ok = None  # 네트워크 상태 캐시

                def generate(self, text, audio_prompt=None):
                    # 새 이벤트 루프 생성 (스레드 안전)
                    loop = asyncio.new_event_loop()
                    try:
                        return loop.run_until_complete(self._generate_async(text))
                    except Exception as e:
                        logger.warning(f"Edge TTS generation failed: {e}")
                        # 네트워크 오류 시 무음 반환하지 않고 예외 발생
                        raise
                    finally:
                        loop.close()

                async def _generate_async(self, text):
                    import io
                    import soundfile as sf

                    communicate = edge_tts.Communicate(text, self.voice)
                    audio_data = b""
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            audio_data += chunk["data"]

                    if not audio_data:
                        raise RuntimeError("Edge TTS returned no audio data")

                    # MP3 -> numpy
                    audio_io = io.BytesIO(audio_data)
                    audio, sr = sf.read(audio_io)
                    # 리샘플링
                    if sr != self.sample_rate:
                        import librosa
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                    return audio.astype(np.float32)

            # Edge TTS 연결 테스트
            test_wrapper = EdgeTTSFallback(self.sample_rate)
            try:
                test_audio = test_wrapper.generate("test")
                if len(test_audio) > 0:
                    logger.info("Using Edge TTS as fallback TTS (high quality)")
                    return test_wrapper
            except Exception as e:
                logger.warning(f"Edge TTS connectivity test failed: {e}")
                raise ImportError("Edge TTS not available (network issue)")

        except (ImportError, Exception) as e:
            logger.warning(f"edge-tts not available ({e}), trying gTTS...")

        try:
            # gTTS 시도
            from gtts import gTTS
            import io
            import soundfile as sf

            class GTTSWrapper:
                def __init__(self, sample_rate):
                    self.sample_rate = sample_rate

                def generate(self, text, audio_prompt=None):
                    tts = gTTS(text=text, lang='ko')
                    mp3_fp = io.BytesIO()
                    tts.write_to_fp(mp3_fp)
                    mp3_fp.seek(0)

                    # MP3 -> numpy (librosa 사용)
                    try:
                        import librosa
                        audio, _ = librosa.load(mp3_fp, sr=self.sample_rate)
                        return audio
                    except ImportError:
                        # soundfile 사용
                        audio, sr = sf.read(mp3_fp)
                        return audio

            logger.info("Using gTTS as fallback TTS")
            return GTTSWrapper(self.sample_rate)

        except ImportError:
            pass

        try:
            # pyttsx3 시도
            import pyttsx3

            class Pyttsx3Wrapper:
                def __init__(self, sample_rate):
                    self.engine = pyttsx3.init()
                    self.sample_rate = sample_rate

                def generate(self, text, audio_prompt=None):
                    # pyttsx3는 직접 오디오 반환이 어려움
                    # 무음 반환
                    duration = len(text) * 0.1
                    return np.zeros(int(duration * self.sample_rate), dtype=np.float32)

            logger.info("Using pyttsx3 as fallback TTS")
            return Pyttsx3Wrapper(self.sample_rate)

        except ImportError:
            pass

        # 최종 폴백: 무음 생성기
        logger.warning("No TTS engine available. Using silent fallback.")
        return SilentTTS(self.sample_rate)

    async def load_voice(
        self,
        voice_path: str,
        voice_id: str = "custom",
    ) -> bool:
        """
        음성 샘플 로드 (음성 클로닝용)

        Args:
            voice_path: 음성 파일 경로
            voice_id: 음성 ID

        Returns:
            로드 성공 여부
        """
        try:
            path = Path(voice_path)
            if not path.exists():
                logger.error(f"Voice file not found: {voice_path}")
                return False

            # Chatterbox는 파일 경로를 직접 사용
            # 다른 TTS는 오디오 데이터를 저장
            self._voice_embeddings[voice_id] = str(path.absolute())

            logger.info(f"Voice loaded: {voice_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load voice: {e}")
            return False

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        텍스트를 음성으로 변환

        Args:
            text: 변환할 텍스트
            voice_id: 사용할 음성 ID (없으면 기본)

        Returns:
            오디오 데이터 (numpy array, float32)
        """
        if not self._initialized:
            await self.initialize()

        if not text.strip():
            return np.array([], dtype=np.float32)

        try:
            # 음성 프롬프트 경로 가져오기 (음성 클로닝용)
            voice_prompt_path = None
            if voice_id and voice_id in self._voice_embeddings:
                voice_prompt_path = self._voice_embeddings[voice_id]

            # 추론 실행
            if hasattr(self._model, 'generate'):
                # Chatterbox API: generate(text, audio_prompt_path=None, ...)
                audio = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._model.generate(
                        text=text,
                        audio_prompt_path=voice_prompt_path
                    )
                )
            else:
                # 폴백
                audio = self._generate_fallback(text)

            # numpy array 확인
            if hasattr(audio, 'cpu'):
                audio = audio.cpu().numpy()

            if len(audio.shape) > 1:
                audio = audio.flatten()

            return audio.astype(np.float32)

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return self._generate_fallback(text)

    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        chunk_size: int = 4096,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        텍스트를 음성으로 스트리밍 변환

        Args:
            text: 변환할 텍스트
            voice_id: 사용할 음성 ID
            chunk_size: 청크 크기 (샘플 수)

        Yields:
            오디오 청크 (numpy array)
        """
        if not self._initialized:
            await self.initialize()

        if not text.strip():
            return

        # 문장 단위로 분리
        sentences = self._split_sentences(text)

        for sentence in sentences:
            if not sentence.strip():
                continue

            # 문장별 합성
            audio = await self.synthesize(sentence, voice_id)

            # 청크로 분할
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                yield chunk

                # 스트리밍 효과를 위한 작은 지연
                await asyncio.sleep(len(chunk) / self.sample_rate * 0.1)

    def _split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분리"""
        import re

        # 한국어/영어 문장 종결 패턴
        pattern = r'(?<=[.!?。！？])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _generate_fallback(self, text: str) -> np.ndarray:
        """폴백 오디오 생성"""
        # 텍스트 길이에 비례한 무음
        duration = len(text) * 0.08  # 글자당 약 80ms
        samples = int(duration * self.sample_rate)
        return np.zeros(samples, dtype=np.float32)

    def get_available_voices(self) -> List[str]:
        """사용 가능한 음성 목록"""
        return list(self._voice_embeddings.keys())

    async def cleanup(self) -> None:
        """리소스 정리"""
        self._model = None
        self._voice_encoder = None
        self._voice_embeddings.clear()
        self._initialized = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass

        logger.info("Chatterbox TTS model cleaned up")


class SilentTTS:
    """무음 TTS (최종 폴백)"""

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def generate(self, text: str, audio_prompt=None) -> np.ndarray:
        duration = len(text) * 0.08
        return np.zeros(int(duration * self.sample_rate), dtype=np.float32)


class EdgeTTSWrapper:
    """
    Edge TTS Wrapper (Microsoft Edge TTS)

    무료 고품질 TTS 대안
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.voice = "ko-KR-SunHiNeural"  # 한국어 여성 음성

    async def generate_async(
        self,
        text: str,
        voice: Optional[str] = None,
    ) -> np.ndarray:
        """비동기 TTS 생성"""
        try:
            import edge_tts
            import io
            import soundfile as sf

            communicate = edge_tts.Communicate(
                text,
                voice or self.voice,
            )

            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            # MP3 -> numpy
            audio_io = io.BytesIO(audio_data)
            audio, sr = sf.read(audio_io)

            # 리샘플링
            if sr != self.sample_rate:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                except ImportError:
                    pass

            return audio.astype(np.float32)

        except ImportError:
            logger.warning("edge-tts not installed")
            return np.zeros(int(len(text) * 0.08 * self.sample_rate), dtype=np.float32)

    def generate(self, text: str, audio_prompt=None) -> np.ndarray:
        """동기 TTS 생성"""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.generate_async(text))
        finally:
            loop.close()
