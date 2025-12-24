"""
ElevenLabs TTS Integration Module.

ElevenLabs TTS API 통합 모듈
- 고품질 음성 합성
- 다국어 지원 (한국어, 영어 등)
- 음성 복제 (Voice Cloning) 지원
- 스트리밍 출력 지원
"""

import asyncio
import io
import logging
from typing import AsyncGenerator, List, Optional, Dict, Any

import numpy as np
import httpx

logger = logging.getLogger(__name__)


class ElevenLabsTTSModel:
    """
    ElevenLabs TTS 모델 래퍼

    Features:
    - 고품질 음성 합성
    - 음성 복제 (Voice Cloning)
    - 다국어 지원
    - 스트리밍 출력
    """

    def __init__(
        self,
        api_key: str,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_multilingual_v2",  # 한국어 지원 모델
        sample_rate: int = 22050,
    ):
        """
        Initialize ElevenLabs TTS Model.

        Args:
            api_key: ElevenLabs API 키
            voice_id: 사용할 음성 ID (None이면 기본 음성 사용)
            model_id: 사용할 모델 ID
            sample_rate: 출력 샘플레이트
        """
        self.api_key = api_key
        self.voice_id = voice_id or "21m00Tcm4TlvDq8ikWAM"  # 기본 음성 (Rachel)
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.base_url = "https://api.elevenlabs.io/v1"
        
        self._initialized = False
        self._client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> bool:
        """모델 초기화"""
        if self._initialized:
            return True

        try:
            # HTTP 클라이언트 생성
            self._client = httpx.AsyncClient(
                timeout=60.0,
                headers={
                    "xi-api-key": self.api_key,
                },
            )

            # API 키 검증 (음성 목록 조회로 확인)
            try:
                response = await self._client.get(f"{self.base_url}/voices")
                if response.status_code == 200:
                    voices = response.json()
                    logger.info(f"✅ ElevenLabs API 연결 성공: {len(voices.get('voices', []))}개 음성 사용 가능")
                    
                    # voice_id가 유효한지 확인
                    if self.voice_id:
                        voice_ids = [v.get("voice_id") for v in voices.get("voices", [])]
                        if self.voice_id not in voice_ids:
                            logger.warning(f"⚠️ 지정된 voice_id '{self.voice_id}'를 찾을 수 없습니다. 기본 음성을 사용합니다.")
                            if voice_ids:
                                self.voice_id = voice_ids[0]
                else:
                    logger.warning(f"⚠️ ElevenLabs API 응답 오류: {response.status_code}")
                    return False
            except Exception as e:
                logger.error(f"❌ ElevenLabs API 연결 실패: {e}")
                return False

            self._initialized = True
            logger.info(f"✅ ElevenLabs TTS 초기화 완료: voice_id={self.voice_id}, model={self.model_id}")
            return True

        except Exception as e:
            logger.error(f"❌ ElevenLabs TTS 초기화 실패: {e}", exc_info=True)
            return False

    async def synthesize(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "ko",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
    ) -> np.ndarray:
        """
        텍스트를 음성으로 변환

        Args:
            text: 변환할 텍스트
            voice_id: 사용할 음성 ID (None이면 기본 음성 사용)
            language: 언어 코드 (ko, en 등)
            stability: 안정성 (0.0 ~ 1.0)
            similarity_boost: 유사도 부스트 (0.0 ~ 1.0)

        Returns:
            오디오 데이터 (numpy array, float32, -1.0 ~ 1.0)
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("ElevenLabs TTS not initialized")

        if not text.strip():
            return np.array([], dtype=np.float32)

        use_voice_id = voice_id or self.voice_id

        try:
            # API 요청
            url = f"{self.base_url}/text-to-speech/{use_voice_id}"
            
            payload = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                },
            }

            logger.debug(f"🎙️ ElevenLabs TTS 요청: text='{text[:50]}...', voice_id={use_voice_id}")

            response = await self._client.post(url, json=payload)
            
            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"❌ ElevenLabs TTS API 오류: {response.status_code} - {error_msg}")
                raise Exception(f"ElevenLabs API error: {response.status_code} - {error_msg}")

            # 오디오 데이터 받기
            audio_bytes = response.content
            
            # MP3/WAV 디코딩
            try:
                import soundfile as sf
                audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            except Exception:
                # soundfile이 실패하면 pydub 사용
                try:
                    from pydub import AudioSegment
                    from pydub.utils import make_chunks
                    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                    audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                    if audio_segment.channels == 2:
                        audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                    audio_data = audio_data / (2 ** 15)  # int16 to float32
                    sr = audio_segment.frame_rate
                except Exception as e:
                    logger.error(f"❌ 오디오 디코딩 실패: {e}")
                    raise

            # 모노로 변환 (스테레오인 경우)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)

            # 샘플레이트 변환
            if sr != self.sample_rate:
                try:
                    import librosa
                    audio_data = librosa.resample(
                        audio_data.astype(np.float32),
                        orig_sr=sr,
                        target_sr=self.sample_rate,
                    )
                except ImportError:
                    logger.warning(f"librosa가 없어 샘플레이트 변환을 건너뜁니다: {sr}Hz → {self.sample_rate}Hz")
                    # 간단한 리샘플링 (선형 보간)
                    from scipy import signal
                    num_samples = int(len(audio_data) * self.sample_rate / sr)
                    audio_data = signal.resample(audio_data, num_samples)

            # 정규화
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            logger.debug(f"✅ ElevenLabs TTS 완료: {len(audio_data)} samples ({len(audio_data)/self.sample_rate:.2f}초)")
            return audio_data

        except Exception as e:
            logger.error(f"❌ ElevenLabs TTS 합성 오류: {e}", exc_info=True)
            raise

    async def synthesize_stream(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "ko",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        chunk_size: int = 4096,
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        텍스트를 음성으로 변환 (스트리밍)

        Args:
            text: 변환할 텍스트
            voice_id: 사용할 음성 ID
            language: 언어 코드
            stability: 안정성
            similarity_boost: 유사도 부스트
            chunk_size: 청크 크기 (샘플 수)

        Yields:
            오디오 청크 (numpy array)
        """
        if not self._initialized or self._client is None:
            raise RuntimeError("ElevenLabs TTS not initialized")

        if not text.strip():
            return

        use_voice_id = voice_id or self.voice_id

        try:
            # 스트리밍 API 요청
            url = f"{self.base_url}/text-to-speech/{use_voice_id}/stream"
            
            payload = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                },
            }

            logger.debug(f"🎙️ ElevenLabs TTS 스트리밍 요청: text='{text[:50]}...'")

            async with self._client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_msg = await response.aread()
                    logger.error(f"❌ ElevenLabs TTS 스트리밍 오류: {response.status_code} - {error_msg}")
                    raise Exception(f"ElevenLabs API error: {response.status_code}")

                # 오디오 청크 스트리밍
                audio_buffer = b""
                async for chunk in response.aiter_bytes():
                    audio_buffer += chunk
                    
                    # 충분한 데이터가 모이면 디코딩하여 반환
                    if len(audio_buffer) >= chunk_size * 2:  # 16-bit = 2 bytes per sample
                        try:
                            import soundfile as sf
                            audio_data, sr = sf.read(io.BytesIO(audio_buffer))
                            
                            # 모노로 변환
                            if len(audio_data.shape) > 1:
                                audio_data = np.mean(audio_data, axis=1)
                            
                            # 샘플레이트 변환
                            if sr != self.sample_rate:
                                import librosa
                                audio_data = librosa.resample(
                                    audio_data.astype(np.float32),
                                    orig_sr=sr,
                                    target_sr=self.sample_rate,
                                )
                            
                            # 정규화
                            audio_data = audio_data.astype(np.float32)
                            if np.max(np.abs(audio_data)) > 1.0:
                                audio_data = audio_data / np.max(np.abs(audio_data))
                            
                            # 청크로 분할
                            for i in range(0, len(audio_data), chunk_size):
                                chunk = audio_data[i:i + chunk_size]
                                if len(chunk) > 0:
                                    yield chunk
                            
                            audio_buffer = b""
                        except Exception as e:
                            logger.debug(f"오디오 디코딩 중 오류 (계속 진행): {e}")
                            continue

                # 남은 버퍼 처리
                if len(audio_buffer) > 0:
                    try:
                        import soundfile as sf
                        audio_data, sr = sf.read(io.BytesIO(audio_buffer))
                        
                        if len(audio_data.shape) > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        
                        if sr != self.sample_rate:
                            import librosa
                            audio_data = librosa.resample(
                                audio_data.astype(np.float32),
                                orig_sr=sr,
                                target_sr=self.sample_rate,
                            )
                        
                        audio_data = audio_data.astype(np.float32)
                        if np.max(np.abs(audio_data)) > 1.0:
                            audio_data = audio_data / np.max(np.abs(audio_data))
                        
                        yield audio_data
                    except Exception as e:
                        logger.debug(f"마지막 오디오 디코딩 오류: {e}")

        except Exception as e:
            logger.error(f"❌ ElevenLabs TTS 스트리밍 오류: {e}", exc_info=True)
            raise

    async def synthesize_sentences_streaming(
        self,
        text: str,
        voice_id: Optional[str] = None,
        language: str = "ko",
    ) -> AsyncGenerator[tuple, None]:
        """
        실시간 문장 단위 TTS 스트리밍 (실제 스트리밍 API 사용)

        ElevenLabs의 스트리밍 API를 사용하여 실시간으로 오디오 청크를 받아서 전송합니다.
        첫 번째 오디오 청크를 빠르게 받을 수 있어 지연시간이 크게 줄어듭니다.

        Args:
            text: 변환할 전체 텍스트
            voice_id: 사용할 음성 ID
            language: 언어 코드

        Yields:
            (audio_array, sentence_text, sentence_index, total_sentences) 튜플
            - audio_array: 오디오 청크 (numpy array)
            - sentence_text: 현재 처리 중인 문장 텍스트
            - sentence_index: 현재 문장 인덱스
            - total_sentences: 전체 문장 수
        """
        if not self._initialized:
            await self.initialize()

        if not text.strip():
            return

        if not self._client:
            raise RuntimeError("ElevenLabs TTS not initialized")

        # 문장 분할
        import re
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        total = len(sentences)

        if total == 0:
            return

        use_voice_id = voice_id or self.voice_id

        logger.info(f"🎙️ ElevenLabs 실시간 TTS 스트리밍 시작: {total}개 문장 (스트리밍 모드)")

        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            try:
                # 스트리밍 API 요청
                url = f"{self.base_url}/text-to-speech/{use_voice_id}/stream"
                
                payload = {
                    "text": sentence,
                    "model_id": self.model_id,
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                }

                logger.debug(f"🎙️ 문장 {idx+1}/{total} 스트리밍 시작: '{sentence[:30]}...'")

                # 첫 번째 청크를 빠르게 받기 위한 스트리밍
                first_chunk_received = False
                audio_chunks = []

                async with self._client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        error_msg = await response.aread()
                        logger.error(f"❌ ElevenLabs TTS 스트리밍 오류: {response.status_code} - {error_msg}")
                        raise Exception(f"ElevenLabs API error: {response.status_code}")

                    # 오디오 청크를 실시간으로 받아서 처리
                    audio_buffer = b""
                    chunk_size_samples = 4096  # 샘플 단위 청크 크기
                    
                    async for chunk_bytes in response.aiter_bytes():
                        audio_buffer += chunk_bytes
                        
                        # MP3/WAV 디코딩 시도 (충분한 데이터가 모였을 때)
                        # MP3는 최소 프레임 크기가 필요하므로 버퍼가 충분히 쌓일 때까지 대기
                        if len(audio_buffer) >= 8192:  # 최소 디코딩 가능한 크기
                            try:
                                # soundfile로 디코딩 시도
                                try:
                                    import soundfile as sf
                                    audio_data, sr = sf.read(io.BytesIO(audio_buffer))
                                    
                                    # 모노로 변환
                                    if len(audio_data.shape) > 1:
                                        audio_data = np.mean(audio_data, axis=1)
                                    
                                    # 샘플레이트 변환
                                    if sr != self.sample_rate:
                                        import librosa
                                        audio_data = librosa.resample(
                                            audio_data.astype(np.float32),
                                            orig_sr=sr,
                                            target_sr=self.sample_rate,
                                        )
                                    
                                    # 정규화
                                    audio_data = audio_data.astype(np.float32)
                                    if np.max(np.abs(audio_data)) > 1.0:
                                        audio_data = audio_data / np.max(np.abs(audio_data))
                                    
                                    # 청크로 분할하여 즉시 전송
                                    processed_samples = 0
                                    while processed_samples < len(audio_data):
                                        chunk_end = min(processed_samples + chunk_size_samples, len(audio_data))
                                        chunk = audio_data[processed_samples:chunk_end]
                                        
                                        if len(chunk) > 0:
                                            # 첫 청크인 경우 로그
                                            if not first_chunk_received:
                                                first_chunk_received = True
                                                logger.info(f"🎵 문장 {idx+1}/{total} 첫 오디오 청크 수신 (지연시간 최소화)")
                                            
                                            yield (chunk, sentence, idx, total)
                                            audio_chunks.append(chunk)
                                        
                                        processed_samples = chunk_end
                                    
                                    # 처리된 데이터는 버퍼에서 제거
                                    # 실제로는 전체 버퍼를 처리했으므로 초기화
                                    audio_buffer = b""
                                    
                                except Exception as sf_error:
                                    # soundfile 실패 시 pydub 시도
                                    try:
                                        from pydub import AudioSegment
                                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_buffer))
                                        audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                                        if audio_segment.channels == 2:
                                            audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                                        audio_data = audio_data / (2 ** 15)  # int16 to float32
                                        sr = audio_segment.frame_rate
                                        
                                        # 샘플레이트 변환
                                        if sr != self.sample_rate:
                                            import librosa
                                            audio_data = librosa.resample(
                                                audio_data.astype(np.float32),
                                                orig_sr=sr,
                                                target_sr=self.sample_rate,
                                            )
                                        
                                        # 정규화
                                        audio_data = audio_data.astype(np.float32)
                                        if np.max(np.abs(audio_data)) > 1.0:
                                            audio_data = audio_data / np.max(np.abs(audio_data))
                                        
                                        # 청크로 분할
                                        processed_samples = 0
                                        while processed_samples < len(audio_data):
                                            chunk_end = min(processed_samples + chunk_size_samples, len(audio_data))
                                            chunk = audio_data[processed_samples:chunk_end]
                                            
                                            if len(chunk) > 0:
                                                if not first_chunk_received:
                                                    first_chunk_received = True
                                                    logger.info(f"🎵 문장 {idx+1}/{total} 첫 오디오 청크 수신")
                                                
                                                yield (chunk, sentence, idx, total)
                                                audio_chunks.append(chunk)
                                            
                                            processed_samples = chunk_end
                                        
                                        audio_buffer = b""
                                        
                                    except Exception as pydub_error:
                                        # 디코딩 실패 시 버퍼 유지하고 계속 수집
                                        logger.debug(f"오디오 디코딩 대기 중... (버퍼: {len(audio_buffer)} bytes)")
                                        continue
                                        
                            except Exception as e:
                                logger.debug(f"오디오 디코딩 중 오류 (계속 진행): {e}")
                                continue

                # 남은 버퍼 처리
                if len(audio_buffer) > 0:
                    try:
                        import soundfile as sf
                        audio_data, sr = sf.read(io.BytesIO(audio_buffer))
                        
                        if len(audio_data.shape) > 1:
                            audio_data = np.mean(audio_data, axis=1)
                        
                        if sr != self.sample_rate:
                            import librosa
                            audio_data = librosa.resample(
                                audio_data.astype(np.float32),
                                orig_sr=sr,
                                target_sr=self.sample_rate,
                            )
                        
                        audio_data = audio_data.astype(np.float32)
                        if np.max(np.abs(audio_data)) > 1.0:
                            audio_data = audio_data / np.max(np.abs(audio_data))
                        
                        # 마지막 청크 전송
                        if len(audio_data) > 0:
                            if not first_chunk_received:
                                first_chunk_received = True
                                logger.info(f"🎵 문장 {idx+1}/{total} 첫 오디오 청크 수신")
                            
                            yield (audio_data, sentence, idx, total)
                            audio_chunks.append(audio_data)
                    except Exception as e:
                        logger.debug(f"마지막 오디오 디코딩 오류: {e}")

                # 문장 완료 로그
                total_audio_length = sum(len(chunk) for chunk in audio_chunks)
                audio_duration = total_audio_length / self.sample_rate if total_audio_length > 0 else 0
                logger.info(
                    f"✅ 문장 {idx+1}/{total} TTS 완료: "
                    f"'{sentence[:30]}...' → {audio_duration:.1f}초 오디오, "
                    f"{len(audio_chunks)}개 청크"
                )

            except Exception as e:
                logger.error(f"❌ 문장 {idx+1} TTS 오류: {e}", exc_info=True)
                # 오류 발생 시 빈 오디오 반환
                empty_audio = np.array([], dtype=np.float32)
                yield (empty_audio, sentence, idx, total)

    async def get_voices(self) -> List[Dict[str, Any]]:
        """사용 가능한 음성 목록 조회"""
        if not self._initialized or self._client is None:
            await self.initialize()

        try:
            response = await self._client.get(f"{self.base_url}/voices")
            if response.status_code == 200:
                data = response.json()
                return data.get("voices", [])
            else:
                logger.error(f"음성 목록 조회 실패: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"음성 목록 조회 오류: {e}")
            return []

    async def cleanup(self) -> None:
        """리소스 정리"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._initialized = False
        logger.info("ElevenLabs TTS cleaned up")

