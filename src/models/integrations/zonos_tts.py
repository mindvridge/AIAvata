"""
Zonos TTS Integration Module.

Zonos TTS - 고품질 다국어 음성 합성 및 음성 복제
- 지원 언어: Korean, English, Japanese, Chinese, French, German
- 음성 복제: 몇 초의 참조 오디오로 가능
- 감정 제어: 행복, 슬픔, 분노, 두려움 등
"""

# torch.compile() 설정 (플랫폼별 최적화)
# - Windows: Triton 미지원으로 비활성화
# - Linux: 환경변수 ZONOS_ENABLE_COMPILE=1 설정 시 활성화 (속도 향상)
import os
import platform

if platform.system() == "Windows":
    # Windows에서는 Triton 미지원으로 torch.compile() 비활성화
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
elif os.environ.get("ZONOS_ENABLE_COMPILE", "0") != "1":
    # Linux/Mac에서도 기본적으로 비활성화 (안정성)
    # ZONOS_ENABLE_COMPILE=1 설정 시 활성화하여 속도 향상 가능
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
else:
    # ZONOS_ENABLE_COMPILE=1 설정 시 torch.compile() 활성화
    pass  # torch.compile() 사용

import asyncio
import io
import json
import logging
import shutil
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """음성 프로필 데이터 클래스"""
    id: str
    name: str
    description: str
    language: str
    audio_path: str
    embedding_path: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    duration_seconds: float = 0.0
    sample_rate: int = 44100

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


class ZonosTTSModel:
    """
    Zonos TTS 모델 래퍼

    Features:
    - 고품질 음성 합성
    - 음성 복제 (참조 오디오 기반)
    - 감정 제어
    - 다국어 지원 (한국어, 영어, 일본어, 중국어, 프랑스어, 독일어)
    """

    # 지원 언어 (ko: 비공식 지원이지만 성능 우수)
    SUPPORTED_LANGUAGES = ["ko", "en", "ja", "zh", "fr", "de"]

    # 감정 매핑
    EMOTIONS = {
        "neutral": 0,
        "happy": 1,
        "sad": 2,
        "angry": 3,
        "fear": 4,
        "surprise": 5,
        "disgust": 6,
    }

    def __init__(
        self,
        model_name: str = "Zyphra/Zonos-v0.1-transformer",
        device: str = "cuda",
        sample_rate: int = 44100,
        voices_dir: str = "assets/voices",
    ):
        """
        Initialize Zonos TTS Model.

        Args:
            model_name: 모델 이름 (transformer 또는 hybrid)
            device: 연산 장치 (cuda, cpu)
            sample_rate: 출력 샘플레이트 (Zonos: 44100Hz)
            voices_dir: 음성 프로필 저장 디렉토리
        """
        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self.voices_dir = Path(voices_dir)

        self._model = None
        self._initialized = False
        self._voice_profiles: Dict[str, VoiceProfile] = {}
        self._speaker_embeddings: Dict[str, Any] = {}

        # 디렉토리 생성
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        (self.voices_dir / "audio").mkdir(exist_ok=True)
        (self.voices_dir / "embeddings").mkdir(exist_ok=True)

        logger.info(f"ZonosTTS initialized: model={model_name}, device={device}")

    async def initialize(self) -> bool:
        """모델 초기화"""
        if self._initialized:
            return True

        try:
            logger.info("Loading Zonos TTS model...")

            # Zonos 모델 로드
            # NOTE: TORCHDYNAMO_DISABLE=1이 파일 상단에서 설정되어 torch.compile() 비활성화됨
            try:
                from zonos.model import Zonos

                self._model = Zonos.from_pretrained(self.model_name, device=self.device)
                logger.info(f"Zonos model loaded successfully on {self.device}")

            except ImportError as e:
                logger.warning(f"Zonos not installed: {e}")
                logger.info("Install with: pip install zonos")
                # Mock 모드로 계속
                self._model = None

            # 저장된 음성 프로필 로드
            await self._load_voice_profiles()

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Zonos TTS: {e}", exc_info=True)
            self._initialized = True  # 폴백 모드
            return False

    async def _load_voice_profiles(self) -> None:
        """저장된 음성 프로필 로드"""
        profiles_file = self.voices_dir / "profiles.json"

        if profiles_file.exists():
            try:
                with open(profiles_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for profile_data in data.get("profiles", []):
                    profile = VoiceProfile(**profile_data)
                    self._voice_profiles[profile.id] = profile

                    # 임베딩 로드
                    if profile.embedding_path and Path(profile.embedding_path).exists():
                        try:
                            import torch
                            embedding = torch.load(profile.embedding_path, map_location=self.device)
                            self._speaker_embeddings[profile.id] = embedding
                        except Exception as e:
                            logger.warning(f"Failed to load embedding for {profile.id}: {e}")

                logger.info(f"Loaded {len(self._voice_profiles)} voice profiles")

            except Exception as e:
                logger.error(f"Failed to load voice profiles: {e}")

    async def _save_voice_profiles(self) -> None:
        """음성 프로필 저장"""
        profiles_file = self.voices_dir / "profiles.json"

        try:
            data = {
                "profiles": [asdict(p) for p in self._voice_profiles.values()],
                "updated_at": datetime.now().isoformat(),
            }

            with open(profiles_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Failed to save voice profiles: {e}")

    async def create_voice_profile(
        self,
        voice_id: str,
        name: str,
        audio_data: bytes,
        description: str = "",
        language: str = "en",
    ) -> Optional[VoiceProfile]:
        """
        새 음성 프로필 생성 (음성 복제)

        Args:
            voice_id: 고유 음성 ID
            name: 음성 이름
            audio_data: 참조 오디오 데이터 (WAV/MP3)
            description: 설명
            language: 언어 코드

        Returns:
            생성된 VoiceProfile 또는 None
        """
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Language {language} not supported. Using 'en'.")
            language = "en"

        try:
            # 오디오 파일 저장
            audio_path = self.voices_dir / "audio" / f"{voice_id}.wav"

            # 오디오 데이터 처리 및 저장
            duration = await self._save_audio_file(audio_data, audio_path)

            # 스피커 임베딩 생성
            embedding_path = None
            if self._model is not None:
                try:
                    import torch
                    import torchaudio

                    # 오디오 로드
                    wav, sr = torchaudio.load(str(audio_path))

                    # 스피커 임베딩 생성
                    speaker_embedding = self._model.make_speaker_embedding(wav, sr)

                    # 임베딩 저장
                    embedding_path = str(self.voices_dir / "embeddings" / f"{voice_id}.pt")
                    torch.save(speaker_embedding, embedding_path)

                    self._speaker_embeddings[voice_id] = speaker_embedding

                    logger.info(f"Created speaker embedding for {voice_id}")

                except Exception as e:
                    logger.error(f"Failed to create speaker embedding: {e}")

            # 프로필 생성
            profile = VoiceProfile(
                id=voice_id,
                name=name,
                description=description,
                language=language,
                audio_path=str(audio_path),
                embedding_path=embedding_path,
                duration_seconds=duration,
                sample_rate=self.sample_rate,
            )

            self._voice_profiles[voice_id] = profile
            await self._save_voice_profiles()

            logger.info(f"Created voice profile: {voice_id} ({name})")
            return profile

        except Exception as e:
            logger.error(f"Failed to create voice profile: {e}", exc_info=True)
            return None

    async def _save_audio_file(self, audio_data: bytes, output_path: Path) -> float:
        """오디오 데이터를 WAV 파일로 저장하고 길이 반환"""
        try:
            from pydub import AudioSegment

            # 오디오 포맷 자동 감지
            audio = AudioSegment.from_file(io.BytesIO(audio_data))

            # WAV로 변환 및 저장
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_channels(1)
            audio.export(str(output_path), format="wav")

            return len(audio) / 1000.0  # 밀리초 -> 초

        except ImportError:
            # pydub 없으면 직접 저장
            with open(output_path, "wb") as f:
                f.write(audio_data)
            return 0.0

    async def update_voice_profile(
        self,
        voice_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Optional[VoiceProfile]:
        """음성 프로필 업데이트"""
        if voice_id not in self._voice_profiles:
            logger.warning(f"Voice profile not found: {voice_id}")
            return None

        profile = self._voice_profiles[voice_id]

        if name is not None:
            profile.name = name
        if description is not None:
            profile.description = description
        if language is not None and language in self.SUPPORTED_LANGUAGES:
            profile.language = language

        profile.updated_at = datetime.now().isoformat()

        await self._save_voice_profiles()

        logger.info(f"Updated voice profile: {voice_id}")
        return profile

    async def delete_voice_profile(self, voice_id: str) -> bool:
        """음성 프로필 삭제"""
        if voice_id not in self._voice_profiles:
            logger.warning(f"Voice profile not found: {voice_id}")
            return False

        profile = self._voice_profiles[voice_id]

        try:
            # 파일 삭제
            if profile.audio_path and Path(profile.audio_path).exists():
                os.remove(profile.audio_path)
            if profile.embedding_path and Path(profile.embedding_path).exists():
                os.remove(profile.embedding_path)

            # 메모리에서 제거
            del self._voice_profiles[voice_id]
            if voice_id in self._speaker_embeddings:
                del self._speaker_embeddings[voice_id]

            await self._save_voice_profiles()

            logger.info(f"Deleted voice profile: {voice_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete voice profile: {e}")
            return False

    def get_voice_profile(self, voice_id: str) -> Optional[VoiceProfile]:
        """음성 프로필 조회"""
        return self._voice_profiles.get(voice_id)

    def list_voice_profiles(self) -> List[VoiceProfile]:
        """모든 음성 프로필 목록"""
        return list(self._voice_profiles.values())

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en",
        emotion: str = "neutral",
        speaking_rate: float = 1.0,
        pitch_std: float = 20.0,
    ) -> np.ndarray:
        """
        텍스트를 음성으로 변환

        Args:
            text: 변환할 텍스트
            voice_id: 사용할 음성 프로필 ID
            language: 언어 코드
            emotion: 감정 (neutral, happy, sad, angry, fear, surprise, disgust)
            speaking_rate: 말하기 속도 (0.5 ~ 2.0)
            pitch_std: 피치 변동성

        Returns:
            오디오 데이터 (numpy array, float32, -1.0 ~ 1.0)
        """
        if not self._initialized:
            await self.initialize()

        if not text.strip():
            return np.array([], dtype=np.float32)

        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Language {language} not supported, using 'en'")
            language = "en"

        # 텍스트 길이 및 예상 생성 시간 로깅
        text_len = len(text)
        # 한국어: 약 4자/초, 영어: 약 10자/초 (보수적)
        chars_per_sec = 4 if language == "ko" else 10
        estimated_audio_duration = min(15.0, max(2.0, (text_len / chars_per_sec) * 1.2))
        estimated_tokens = int(86 * estimated_audio_duration)
        # RTX 4070 Ti 기준 약 10 it/s (실측 기반)
        estimated_gen_time = estimated_tokens / 10
        logger.info(f"🎤 TTS 요청: {text_len}자 → 최대 {estimated_audio_duration:.1f}초 오디오, 예상 생성≈{estimated_gen_time:.1f}초")
        logger.debug(f"TTS 텍스트 내용: '{text[:100]}{'...' if len(text) > 100 else ''}'")

        try:
            if self._model is None:
                logger.warning("Zonos model not loaded, using mock audio")
                return self._generate_mock_audio(len(text))

            import torch
            from zonos.conditioning import make_cond_dict

            # 스피커 임베딩 가져오기
            speaker_embedding = None
            if voice_id and voice_id in self._speaker_embeddings:
                speaker_embedding = self._speaker_embeddings[voice_id]

            # 언어 코드 변환 (Zonos 형식)
            lang_map = {
                "ko": "ko",
                "en": "en-us",
                "ja": "ja",
                "zh": "zh",
                "fr": "fr-fr",
                "de": "de",
            }
            zonos_lang = lang_map.get(language, "en-us")

            # 🚀 텍스트 길이 기반 max_new_tokens 계산 (속도 최적화)
            # Zonos: 86 tokens ≈ 1초 오디오
            # 한국어: 약 4자/초 (보수적), 영어: 약 10자/초
            chars_per_second = 4 if language == "ko" else 10
            estimated_duration = len(text) / chars_per_second
            # 🚀 속도 최적화: 최소 2초, 최대 6초로 제한 (짧은 문장 권장)
            # 긴 문장은 문장 단위로 분리되어 처리됨
            max_duration = min(6.0, max(2.0, estimated_duration * 1.2))
            max_new_tokens = int(86 * max_duration)

            logger.info(f"🎯 TTS 토큰 제한: {max_new_tokens} tokens (최대 {max_duration:.1f}초 오디오)")

            # 🎯 무거운 연산을 별도 스레드에서 실행 (이벤트 루프 블로킹 방지)
            # TTS 생성 중에도 idle 루프가 계속 재생될 수 있도록 함
            def _generate_audio():
                import time as time_module
                import torch  # torch import 추가

                # 조건부 생성 (Zonos API)
                cond_dict = make_cond_dict(
                    text=text,
                    speaker=speaker_embedding,
                    language=zonos_lang,
                )

                # 조건부 준비
                conditioning = self._model.prepare_conditioning(cond_dict)

                gen_start = time_module.time()
                with torch.no_grad():
                    # max_new_tokens로 생성 길이 제한 (속도 향상!)
                    codes = self._model.generate(
                        conditioning,
                        max_new_tokens=max_new_tokens,
                    )
                    gen_time = time_module.time() - gen_start
                    logger.info(f"🎵 코드 생성 완료: {codes.shape[-1]} tokens, {gen_time:.2f}초 소요")

                    decode_start = time_module.time()
                    try:
                        logger.info(f"🔊 오디오 디코딩 시작: codes shape={codes.shape}")
                        
                        # GPU 메모리 정리 (디코딩 전)
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                logger.debug("GPU 캐시 정리 완료 (디코딩 전)")
                        except:
                            pass
                        
                        # 디코딩 실행
                        audio = self._model.autoencoder.decode(codes)
                        
                        decode_time = time_module.time() - decode_start
                        logger.info(f"🔊 오디오 디코딩 완료: {decode_time:.2f}초 소요, audio shape={audio.shape if hasattr(audio, 'shape') else 'unknown'}")
                    except Exception as decode_error:
                        decode_time = time_module.time() - decode_start
                        logger.error(f"❌ 오디오 디코딩 실패: {decode_error} (소요 시간: {decode_time:.2f}초)", exc_info=True)
                        # GPU 메모리 정리 (오류 후)
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except:
                            pass
                        raise

                # numpy 변환
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()

                total_time = time_module.time() - gen_start
                return audio, total_time

            # run_in_executor로 비동기 실행 (idle 루프가 계속 실행될 수 있도록)
            loop = asyncio.get_event_loop()
            try:
                # GPU 메모리 확인 (CUDA 사용 시)
                try:
                    import torch
                    if torch.cuda.is_available():
                        gpu_memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                        gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                        logger.info(f"💾 GPU 메모리 상태: 할당={gpu_memory_allocated:.2f}GB, 예약={gpu_memory_reserved:.2f}GB")
                except ImportError:
                    pass  # torch가 없으면 무시
                
                audio, total_time = await asyncio.wait_for(
                    loop.run_in_executor(None, _generate_audio),
                    timeout=300.0  # 5분 타임아웃
                )
            except asyncio.TimeoutError:
                logger.error("❌ TTS 생성 타임아웃 (5분 초과)")
                raise Exception("TTS generation timeout")
            except Exception as e:
                logger.error(f"❌ TTS 생성 중 오류: {e}", exc_info=True)
                raise

            # 정규화
            if audio.ndim > 1:
                audio = audio.squeeze()

            audio = audio.astype(np.float32)
            if np.abs(audio).max() > 1.0:
                audio = audio / np.abs(audio).max()

            # Zonos 샘플레이트 사용 (44100Hz)
            self.sample_rate = self._model.autoencoder.sampling_rate
            audio_duration = len(audio) / self.sample_rate

            logger.info(f"✅ TTS 완료: {len(audio)} samples ({audio_duration:.2f}초 오디오) @ {self.sample_rate}Hz, 총 {total_time:.2f}초 소요")
            return audio

        except Exception as e:
            logger.error(f"TTS synthesis error: {e}", exc_info=True)
            return self._generate_mock_audio(len(text))

    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en",
        chunk_size: int = 4096,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        스트리밍 TTS 합성

        현재 Zonos는 전체 합성 후 청크 분할 방식
        """
        audio = await self.synthesize(text, voice_id, language)

        for i in range(0, len(audio), chunk_size):
            yield audio[i:i + chunk_size]
            await asyncio.sleep(0.01)  # 약간의 지연

    def _split_into_sentences(self, text: str, language: str = "ko") -> List[str]:
        """
        텍스트를 문장 단위로 분할

        Args:
            text: 분할할 텍스트
            language: 언어 코드

        Returns:
            문장 리스트
        """
        import re

        if not text.strip():
            return []

        # 한국어/일본어/중국어: 마침표, 물음표, 느낌표로 분할
        if language in ["ko", "ja", "zh"]:
            # 문장 끝 패턴: 마침표, 물음표, 느낌표 + 공백 또는 끝
            sentences = re.split(r'(?<=[.?!。？！])\s*', text)
        else:
            # 영어 등: 마침표, 물음표, 느낌표 뒤 공백으로 분할
            sentences = re.split(r'(?<=[.?!])\s+', text)

        # 빈 문장 제거 및 정리
        sentences = [s.strip() for s in sentences if s.strip()]

        # 너무 짧은 문장은 다음 문장과 합치기 (최소 10자)
        merged = []
        buffer = ""
        for s in sentences:
            if buffer:
                buffer = buffer + " " + s
            else:
                buffer = s

            if len(buffer) >= 10 or s == sentences[-1]:
                merged.append(buffer)
                buffer = ""

        if buffer:
            merged.append(buffer)

        logger.debug(f"Split text into {len(merged)} sentences: {[s[:20]+'...' for s in merged]}")
        return merged

    async def synthesize_sentences_streaming(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "en",
    ) -> AsyncGenerator[tuple[np.ndarray, str, int, int], None]:
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
        if not self._initialized:
            await self.initialize()

        if not text.strip():
            return

        # 문장 분할
        sentences = self._split_into_sentences(text, language)
        total = len(sentences)

        if total == 0:
            return

        logger.info(f"🎙️ 실시간 TTS 스트리밍 시작: {total}개 문장")

        for idx, sentence in enumerate(sentences):
            start_time = asyncio.get_event_loop().time()
            
            try:
                # 개별 문장 TTS 생성 (타임아웃 적용)
                logger.info(f"🎤 문장 {idx+1}/{total} TTS 생성 시작: '{sentence[:30]}...'")
                
                audio = await asyncio.wait_for(
                    self.synthesize(
                        text=sentence,
                        voice_id=voice_id,
                        language=language,
                    ),
                    timeout=120.0  # 2분 타임아웃
                )

                elapsed = asyncio.get_event_loop().time() - start_time
                audio_duration = len(audio) / self.sample_rate if len(audio) > 0 else 0

                logger.info(
                    f"🎵 문장 {idx+1}/{total} TTS 완료: "
                    f"'{sentence[:30]}...' → {audio_duration:.1f}초 오디오, "
                    f"{elapsed:.1f}초 소요"
                )

                yield (audio, sentence, idx, total)
                
            except asyncio.TimeoutError:
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.error(f"❌ 문장 {idx+1}/{total} TTS 타임아웃: {elapsed:.1f}초 후 중단")
                # 빈 오디오 반환하여 다음 문장으로 진행
                empty_audio = np.array([], dtype=np.float32)
                yield (empty_audio, sentence, idx, total)
                
            except Exception as e:
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.error(f"❌ 문장 {idx+1}/{total} TTS 오류: {e} (소요 시간: {elapsed:.1f}초)", exc_info=True)
                # 빈 오디오 반환하여 다음 문장으로 진행
                empty_audio = np.array([], dtype=np.float32)
                yield (empty_audio, sentence, idx, total)

    def _generate_mock_audio(self, text_length: int) -> np.ndarray:
        """테스트용 mock 오디오 생성"""
        duration_samples = int(text_length * 0.1 * self.sample_rate)
        if duration_samples < self.sample_rate // 2:
            duration_samples = self.sample_rate // 2  # 최소 0.5초

        # 사인파 생성
        t = np.linspace(0, duration_samples / self.sample_rate, duration_samples)
        audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        logger.debug(f"Generated mock audio: {duration_samples} samples")
        return audio

    async def get_voice_audio(self, voice_id: str) -> Optional[bytes]:
        """음성 프로필의 원본 오디오 데이터 반환"""
        profile = self.get_voice_profile(voice_id)
        if not profile or not profile.audio_path:
            return None

        try:
            with open(profile.audio_path, "rb") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read voice audio: {e}")
            return None

    async def cleanup(self) -> None:
        """리소스 정리"""
        self._model = None
        self._speaker_embeddings.clear()
        self._initialized = False
        logger.info("Zonos TTS cleaned up")
