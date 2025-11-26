"""
TTS Cache Service for audio caching.

TTS 결과 캐싱 서비스
특징:
- 자주 사용되는 문구 사전 캐싱
- Redis 기반 분산 캐시 지원
- 인메모리 캐시 폴백
- LRU 캐시 정책
"""

import asyncio
import hashlib
import logging
from typing import Optional, Dict
from collections import OrderedDict

import numpy as np

logger = logging.getLogger(__name__)


class TTSCacheService:
    """
    TTS 오디오 캐싱 서비스

    Features:
    - 텍스트 기반 캐시 키 생성
    - 인메모리 LRU 캐시
    - Redis 분산 캐시 (선택적)
    - 자주 사용되는 문구 사전 캐싱
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        max_memory_items: int = 1000,
        max_audio_duration_sec: float = 30.0,
    ):
        """
        Initialize TTS Cache Service.

        Args:
            redis_url: Redis 연결 URL (없으면 인메모리만 사용)
            max_memory_items: 인메모리 캐시 최대 항목 수
            max_audio_duration_sec: 캐시 가능한 최대 오디오 길이
        """
        self.redis_url = redis_url
        self.max_memory_items = max_memory_items
        self.max_audio_duration_sec = max_audio_duration_sec

        # 인메모리 LRU 캐시
        self._memory_cache: OrderedDict[str, bytes] = OrderedDict()

        # Redis 클라이언트
        self._redis = None
        self._initialized = False

        # 통계
        self._stats = {
            "hits": 0,
            "misses": 0,
            "memory_items": 0,
        }

    async def initialize(self) -> None:
        """서비스 초기화"""
        if self._initialized:
            return

        logger.info("Initializing TTS Cache Service...")

        if self.redis_url:
            try:
                import redis.asyncio as redis

                self._redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=False,
                )
                await self._redis.ping()
                logger.info("Redis cache connected")

            except ImportError:
                logger.warning("Redis package not installed. Using memory cache only.")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using memory cache only.")
                self._redis = None

        self._initialized = True
        logger.info("TTS Cache Service initialized")

    def _generate_cache_key(
        self,
        text: str,
        voice_id: Optional[str] = None,
        sample_rate: int = 24000,
    ) -> str:
        """
        캐시 키 생성

        Args:
            text: 텍스트
            voice_id: 음성 ID
            sample_rate: 샘플레이트

        Returns:
            캐시 키 (SHA256 해시)
        """
        key_parts = [
            text.strip().lower(),
            str(voice_id or "default"),
            str(sample_rate),
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get(
        self,
        text: str,
        voice_id: Optional[str] = None,
        sample_rate: int = 24000,
    ) -> Optional[bytes]:
        """
        캐시에서 오디오 조회

        Args:
            text: 텍스트
            voice_id: 음성 ID
            sample_rate: 샘플레이트

        Returns:
            캐시된 오디오 bytes 또는 None
        """
        if not self._initialized:
            return None

        key = self._generate_cache_key(text, voice_id, sample_rate)

        # 1. 인메모리 캐시 확인
        if key in self._memory_cache:
            # LRU: 최근 사용 항목을 끝으로 이동
            self._memory_cache.move_to_end(key)
            self._stats["hits"] += 1
            logger.debug(f"Cache hit (memory): {key[:16]}...")
            return self._memory_cache[key]

        # 2. Redis 캐시 확인
        if self._redis:
            try:
                data = await self._redis.get(f"tts:{key}")
                if data:
                    # 인메모리에도 저장 (hot data)
                    self._set_memory(key, data)
                    self._stats["hits"] += 1
                    logger.debug(f"Cache hit (redis): {key[:16]}...")
                    return data
            except Exception as e:
                logger.warning(f"Redis get error: {e}")

        self._stats["misses"] += 1
        return None

    async def set(
        self,
        text: str,
        audio_data: bytes,
        voice_id: Optional[str] = None,
        sample_rate: int = 24000,
        ttl_seconds: int = 3600,
    ) -> bool:
        """
        오디오를 캐시에 저장

        Args:
            text: 텍스트
            audio_data: 오디오 데이터 (bytes)
            voice_id: 음성 ID
            sample_rate: 샘플레이트
            ttl_seconds: Redis TTL (초)

        Returns:
            저장 성공 여부
        """
        if not self._initialized:
            return False

        # 오디오 길이 체크 (너무 긴 것은 캐시 안 함)
        audio_duration = len(audio_data) / (sample_rate * 2)  # 16-bit
        if audio_duration > self.max_audio_duration_sec:
            logger.debug(f"Audio too long for cache: {audio_duration:.1f}s")
            return False

        key = self._generate_cache_key(text, voice_id, sample_rate)

        # 1. 인메모리 캐시 저장
        self._set_memory(key, audio_data)

        # 2. Redis 캐시 저장
        if self._redis:
            try:
                await self._redis.setex(
                    f"tts:{key}",
                    ttl_seconds,
                    audio_data,
                )
                logger.debug(f"Cache set (redis): {key[:16]}...")
            except Exception as e:
                logger.warning(f"Redis set error: {e}")

        return True

    def _set_memory(self, key: str, data: bytes) -> None:
        """인메모리 캐시에 저장 (LRU)"""
        # 이미 있으면 삭제 후 다시 추가 (최근으로)
        if key in self._memory_cache:
            del self._memory_cache[key]

        # 용량 초과 시 가장 오래된 항목 삭제
        while len(self._memory_cache) >= self.max_memory_items:
            self._memory_cache.popitem(last=False)

        self._memory_cache[key] = data
        self._stats["memory_items"] = len(self._memory_cache)

    async def delete(
        self,
        text: str,
        voice_id: Optional[str] = None,
        sample_rate: int = 24000,
    ) -> bool:
        """캐시에서 항목 삭제"""
        key = self._generate_cache_key(text, voice_id, sample_rate)

        # 인메모리 삭제
        if key in self._memory_cache:
            del self._memory_cache[key]
            self._stats["memory_items"] = len(self._memory_cache)

        # Redis 삭제
        if self._redis:
            try:
                await self._redis.delete(f"tts:{key}")
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")

        return True

    async def clear(self) -> None:
        """전체 캐시 삭제"""
        # 인메모리 클리어
        self._memory_cache.clear()
        self._stats["memory_items"] = 0

        # Redis 클리어 (tts: 프리픽스만)
        if self._redis:
            try:
                cursor = 0
                while True:
                    cursor, keys = await self._redis.scan(
                        cursor, match="tts:*", count=100
                    )
                    if keys:
                        await self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")

        logger.info("TTS cache cleared")

    async def preload_common_phrases(
        self,
        phrases: list,
        tts_func,
        voice_id: Optional[str] = None,
        sample_rate: int = 24000,
    ) -> int:
        """
        자주 사용되는 문구 사전 캐싱

        Args:
            phrases: 캐싱할 문구 리스트
            tts_func: TTS 함수 (async callable)
            voice_id: 음성 ID
            sample_rate: 샘플레이트

        Returns:
            캐싱된 항목 수
        """
        cached_count = 0

        for phrase in phrases:
            # 이미 캐시되어 있으면 스킵
            existing = await self.get(phrase, voice_id, sample_rate)
            if existing:
                continue

            try:
                # TTS 생성
                audio = await tts_func(phrase)
                if isinstance(audio, np.ndarray):
                    audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                else:
                    audio_bytes = audio

                # 캐시 저장
                await self.set(phrase, audio_bytes, voice_id, sample_rate)
                cached_count += 1

            except Exception as e:
                logger.warning(f"Failed to preload phrase '{phrase}': {e}")

        logger.info(f"Preloaded {cached_count} common phrases")
        return cached_count

    def get_stats(self) -> Dict:
        """캐시 통계 반환"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        return {
            **self._stats,
            "hit_rate": hit_rate,
            "total_requests": total,
        }

    async def cleanup(self) -> None:
        """리소스 정리"""
        if self._redis:
            await self._redis.close()
            self._redis = None

        self._memory_cache.clear()
        self._initialized = False
        logger.info("TTS Cache Service cleaned up")
