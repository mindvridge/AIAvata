"""
Edge TTS Integration Module.

Google TTS (gTTS) - 무료, 안정적인 음성 합성
- 안정적인 응답
- 다양한 언어 지원
- 한국어 포함 다국어 지원

Note: 원래 Edge TTS였으나 안정성 문제로 gTTS로 전환
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, AsyncGenerator

import numpy as np

logger = logging.getLogger(__name__)


class EdgeTTSModel:
    """
    TTS 모델 래퍼 (gTTS 사용)
    
    Features:
    - 안정적인 클라우드 기반 TTS
    - 무료 사용
    - 다국어 지원
    """
    
    # 언어별 기본 설정
    DEFAULT_LANGS = {
        "ko": "ko",
        "en": "en",
        "ja": "ja",
        "zh": "zh-CN",
    }
    
    def __init__(
        self,
        default_voice: str = "ko-KR-SunHiNeural",
        sample_rate: int = 24000,
    ):
        """
        Initialize TTS Model.
        
        Args:
            default_voice: 기본 음성 (호환성 유지용, 내부적으로 언어만 사용)
            sample_rate: 출력 샘플레이트
        """
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self._initialized = False
        
        logger.info(f"TTS initialized (gTTS): sample_rate={sample_rate}")
    
    async def initialize(self) -> bool:
        """모델 초기화"""
        try:
            from gtts import gTTS
            self._initialized = True
            logger.info("TTS (gTTS) initialized successfully")
            return True
        except ImportError:
            logger.error("gTTS 패키지가 설치되지 않았습니다!")
            logger.error("   설치: pip install gtts")
            return False
    
    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "ko",
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> np.ndarray:
        """
        텍스트를 음성으로 변환
        
        Args:
            text: 변환할 텍스트
            voice: 사용할 음성 (호환성 유지용)
            language: 언어 코드
            rate: 속도 조절 (gTTS에서는 미지원)
            pitch: 피치 조절 (gTTS에서는 미지원)
            
        Returns:
            오디오 배열 (numpy float32, [-1, 1])
        """
        if not self._initialized:
            await self.initialize()
        
        if not text.strip():
            return np.array([], dtype=np.float32)
        
        import tempfile
        import os
        import concurrent.futures
        from gtts import gTTS
        
        # 언어 매핑
        lang = self.DEFAULT_LANGS.get(language, language)
        
        logger.info(f"gTTS: '{text[:30]}...' lang={lang}")
        
        try:
            import time
            start_time = time.time()
            
            # 임시 파일 경로 생성
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # gTTS로 음성 생성 (동기 작업을 스레드에서 실행)
            def run_gtts():
                try:
                    tts = gTTS(text=text, lang=lang, slow=False)
                    tts.save(tmp_path)
                    return True, None
                except Exception as e:
                    return False, str(e)
            
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                success, error = await loop.run_in_executor(pool, run_gtts)
            
            if not success:
                logger.error(f"gTTS 에러: {error}")
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                return np.array([], dtype=np.float32)
            
            # MP3 파일 읽기
            if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
                logger.warning("gTTS가 빈 오디오 파일을 생성했습니다")
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                return np.array([], dtype=np.float32)
            
            with open(tmp_path, "rb") as f:
                audio_data = f.read()
            
            # 임시 파일 삭제
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            elapsed = time.time() - start_time
            logger.info(f"gTTS 완료: {len(audio_data)} bytes, {elapsed:.2f}s")
            
            if not audio_data:
                logger.warning("gTTS가 빈 오디오를 반환했습니다")
                return np.array([], dtype=np.float32)
            
            # MP3 디코딩
            audio_array = self._decode_mp3(audio_data)
            
            # 리샘플링 (gTTS는 24000Hz 출력)
            if self.sample_rate != 24000:
                audio_array = self._resample(audio_array, 24000, self.sample_rate)
            
            return audio_array
            
        except Exception as e:
            logger.error(f"gTTS 에러: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return np.array([], dtype=np.float32)
    
    def _decode_mp3(self, mp3_data: bytes) -> np.ndarray:
        """MP3 데이터를 numpy 배열로 디코딩"""
        try:
            # pydub 사용 시도
            from pydub import AudioSegment
            import io
            
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_data))
            audio = audio.set_channels(1)  # 모노로 변환
            
            # numpy 배열로 변환
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / 32768.0  # int16 -> float32 [-1, 1]
            
            return samples
            
        except ImportError:
            logger.warning("pydub 없음, scipy 사용 시도")
            
        try:
            # scipy 사용 시도 (임시 파일 필요)
            import scipy.io.wavfile as wav
            import tempfile
            import subprocess
            
            # ffmpeg으로 MP3 -> WAV 변환
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_file:
                mp3_file.write(mp3_data)
                mp3_path = mp3_file.name
            
            wav_path = mp3_path.replace(".mp3", ".wav")
            
            # ffmpeg 변환
            subprocess.run([
                "ffmpeg", "-y", "-i", mp3_path,
                "-ar", "24000", "-ac", "1",
                wav_path
            ], capture_output=True, check=True)
            
            # WAV 읽기
            sample_rate, audio = wav.read(wav_path)
            audio = audio.astype(np.float32) / 32768.0
            
            # 임시 파일 정리
            Path(mp3_path).unlink(missing_ok=True)
            Path(wav_path).unlink(missing_ok=True)
            
            return audio
            
        except Exception as e:
            logger.error(f"MP3 디코딩 실패: {e}")
            return np.array([], dtype=np.float32)
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """오디오 리샘플링"""
        if orig_sr == target_sr:
            return audio
        
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # 간단한 리샘플링 (품질 낮음)
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    async def synthesize_sentences_streaming(
        self,
        text: str,
        language: str = "ko",
    ) -> AsyncGenerator[tuple, None]:
        """
        문장 단위로 음성 스트리밍 생성
        
        Args:
            text: 전체 텍스트
            language: 언어 코드
            
        Yields:
            (audio_array, sentence_text, sentence_index, total_sentences)
        """
        # 문장 분리
        sentences = self._split_sentences(text)
        total = len(sentences)
        
        logger.info(f"gTTS 스트리밍 시작: {total}개 문장")
        
        for idx, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            logger.info(f"🎤 문장 {idx+1}/{total} 생성: '{sentence[:30]}...'")
            
            audio = await self.synthesize(sentence, language=language)
            
            if len(audio) > 0:
                yield (audio, sentence, idx, total)
    
    def _split_sentences(self, text: str) -> List[str]:
        """텍스트를 문장으로 분리"""
        import re
        
        # 문장 종결 패턴 (한국어/영어/일본어)
        pattern = r'(?<=[.!?。！？])\s*'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def get_voices(self, language: Optional[str] = None) -> List[dict]:
        """사용 가능한 음성 목록 조회 (gTTS는 언어만 지원)"""
        # gTTS는 음성 선택이 없고 언어만 지원
        supported_langs = [
            {"ShortName": "ko", "Locale": "ko", "Gender": "Female", "Name": "Korean"},
            {"ShortName": "en", "Locale": "en", "Gender": "Female", "Name": "English"},
            {"ShortName": "ja", "Locale": "ja", "Gender": "Female", "Name": "Japanese"},
            {"ShortName": "zh-CN", "Locale": "zh-CN", "Gender": "Female", "Name": "Chinese"},
        ]
        
        if language:
            return [v for v in supported_langs if v["Locale"].startswith(language)]
        return supported_langs
    
    async def cleanup(self) -> None:
        """리소스 정리"""
        self._initialized = False
        logger.info("gTTS cleanup completed")

