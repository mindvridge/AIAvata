"""
Pipeline Orchestrator for the Realtime AI Avatar Service.

전체 아바타 파이프라인 조율 모듈
Flow:
1. 사용자 음성 → STT (+ 감정 추출)
2. 텍스트 → LLM (스트리밍 응답)
3. LLM 응답 → TTS (스트리밍)
4. 오디오 → Avatar Renderer (립싱크 + Idle 루프)
5. 비디오 프레임 → WebRTC 전송
"""

import asyncio
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional, Any
from uuid import UUID

import numpy as np

from ..config import Settings
from ..models.emotion import Emotion, EmotionMapping
from ..models.schemas import (
    AvatarSession,
    PipelineState,
    STTResult,
    VideoFrame,
)
from .stt_module import STTModule
from .llm_module import LLMModule
from .tts_module import TTSModule
from .avatar_renderer import AvatarRenderer

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    전체 아바타 파이프라인 조율

    모든 컴포넌트(STT, LLM, TTS, Avatar)를 통합하여
    실시간 대화형 아바타 응답을 생성합니다.
    """

    def __init__(self, settings: Settings):
        """
        Initialize Pipeline Orchestrator.

        Args:
            settings: 애플리케이션 설정
        """
        self.settings = settings

        # 파이프라인 모듈 초기화
        self.stt = STTModule(
            device=settings.get_device(),
            vad_enabled=True,
            model_path=settings.stt_model_path,
        )

        self.llm = LLMModule(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            provider=settings.llm_provider,
        )

        self.tts = TTSModule(
            voice_sample_path=settings.voice_sample_path,
            sample_rate=settings.tts_sample_rate,
            device=settings.get_device(),
        )

        self.renderer = AvatarRenderer(
            idle_loops_dir=settings.idle_loops_dir,
            avatar_image_path=settings.default_avatar_image,
            output_width=settings.video_width,
            output_height=settings.video_height,
            target_fps=settings.target_fps,
            device=settings.get_device(),
        )

        # 세션 관리
        self._sessions: Dict[UUID, AvatarSession] = {}
        self._initialized = False

        # 성능 메트릭
        self._metrics = {
            "total_requests": 0,
            "avg_latency_ms": 0,
            "last_latency_ms": 0,
        }

    async def initialize(self) -> None:
        """모든 파이프라인 컴포넌트 초기화"""
        if self._initialized:
            return

        logger.info("Initializing Pipeline Orchestrator...")

        # 모든 모듈 병렬 초기화
        await asyncio.gather(
            self.stt.initialize(),
            self.llm.initialize(),
            self.tts.initialize(),
            self.renderer.initialize(),
        )

        self._initialized = True
        logger.info("Pipeline Orchestrator initialized successfully")

    def _ensure_initialized(self) -> None:
        """초기화 확인"""
        if not self._initialized:
            raise RuntimeError(
                "Pipeline not initialized. Call initialize() first."
            )

    def create_session(
        self,
        avatar_id: str = "default",
        system_prompt: Optional[str] = None,
    ) -> AvatarSession:
        """
        새 아바타 세션 생성

        Args:
            avatar_id: 사용할 아바타 ID
            system_prompt: 커스텀 시스템 프롬프트

        Returns:
            생성된 세션
        """
        session = AvatarSession(
            avatar_id=avatar_id,
            current_emotion=Emotion.NEUTRAL,
            pipeline_state=PipelineState.IDLE,
        )

        # 세션별 시스템 프롬프트 저장
        if system_prompt:
            session.conversation_history.append({
                "role": "system",
                "content": system_prompt,
            })

        self._sessions[session.session_id] = session
        logger.info(f"Created session: {session.session_id}")

        return session

    def get_session(self, session_id: UUID) -> Optional[AvatarSession]:
        """세션 조회"""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: UUID) -> bool:
        """세션 삭제"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    async def process_audio_input(
        self,
        audio: bytes,
        session_id: Optional[UUID] = None,
        sample_rate: int = 16000,
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        오디오 입력을 처리하여 비디오 프레임 스트림 반환

        전체 파이프라인:
        Audio → STT → LLM → TTS → Avatar Renderer → Video Frames

        Args:
            audio: 사용자 음성 데이터 (bytes)
            session_id: 세션 ID (없으면 임시 세션 사용)
            sample_rate: 오디오 샘플레이트

        Yields:
            VideoFrame: 렌더링된 비디오 프레임
        """
        self._ensure_initialized()
        start_time = time.time()

        # 세션 가져오기 또는 임시 세션 생성
        session = None
        if session_id:
            session = self.get_session(session_id)
        if session is None:
            session = self.create_session()

        # 상태 업데이트
        session.pipeline_state = PipelineState.LISTENING
        session.total_interactions += 1

        try:
            # 1. STT + 감정 인식
            stt_result = await self._process_stt(audio, sample_rate)

            if not stt_result.text.strip():
                # 인식된 텍스트가 없으면 idle 프레임 스트리밍
                session.pipeline_state = PipelineState.IDLE
                async for frame in self.renderer.render_idle_stream(duration=1.0):
                    yield frame
                return

            logger.info(
                f"STT Result: '{stt_result.text}' "
                f"(emotion: {stt_result.emotion}, confidence: {stt_result.confidence:.2f})"
            )

            # 2. 감정에 따른 아바타 상태 설정
            avatar_emotion = EmotionMapping.get_avatar_response(stt_result.emotion)
            self.renderer.set_emotion(avatar_emotion)
            session.current_emotion = avatar_emotion

            # 3. 대화 히스토리에 사용자 메시지 추가
            session.conversation_history.append({
                "role": "user",
                "content": stt_result.text,
            })

            # 4. LLM 스트리밍 → TTS → Avatar 렌더링 파이프라인
            session.pipeline_state = PipelineState.PROCESSING

            # LLM 응답 수집 (대화 히스토리용)
            full_response = ""

            async for video_frame in self._process_llm_tts_render(
                user_text=stt_result.text,
                user_emotion=stt_result.emotion.value,
                conversation_history=session.conversation_history[:-1],  # 현재 메시지 제외
            ):
                yield video_frame

            # 5. 대화 히스토리에 어시스턴트 응답 추가
            # Note: 실제 응답 텍스트는 스트리밍 중 수집해야 함
            session.pipeline_state = PipelineState.IDLE

            # 성능 메트릭 업데이트
            latency = (time.time() - start_time) * 1000
            self._update_metrics(latency)
            logger.info(f"Total pipeline latency: {latency:.2f}ms")

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            session.pipeline_state = PipelineState.ERROR
            raise

        finally:
            session.pipeline_state = PipelineState.IDLE

    async def _process_stt(
        self, audio: bytes, sample_rate: int
    ) -> STTResult:
        """STT 처리"""
        # bytes → numpy array
        audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        return await self.stt.transcribe(
            audio=audio_array,
            sample_rate=sample_rate,
            language=self.settings.default_language,
        )

    async def _process_llm_tts_render(
        self,
        user_text: str,
        user_emotion: str,
        conversation_history: List[dict],
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        LLM → TTS → Avatar 렌더링 파이프라인

        LLM 스트리밍 출력을 TTS로 변환하고,
        오디오에 맞춰 립싱크된 비디오 프레임을 생성합니다.
        """
        # LLM 스트리밍 응답 생성
        llm_stream = self.llm.generate_stream(
            user_message=user_text,
            system_prompt=self.settings.system_prompt,
            conversation_history=conversation_history,
            user_emotion=user_emotion,
        )

        # LLM 텍스트 → TTS 텍스트 스트림으로 변환
        async def text_stream():
            async for response in llm_stream:
                if response.text:
                    yield response.text

        # TTS 스트리밍
        tts_stream = self.tts.synthesize_stream_realtime(
            text_stream=text_stream(),
            chunk_size=self.settings.tts_chunk_size,
        )

        # TTS 오디오 → 비디오 프레임 스트림으로 변환
        async def audio_bytes_stream():
            async for tts_chunk in tts_stream:
                yield tts_chunk.audio_data

        # Avatar 렌더링
        async for frame in self.renderer.render_with_audio(
            audio_stream=audio_bytes_stream(),
            audio_sample_rate=self.settings.tts_sample_rate,
        ):
            yield frame

    async def stream_idle(
        self,
        session_id: Optional[UUID] = None,
        duration: float = -1,
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        Idle 상태 비디오 스트리밍

        대화가 없을 때 자연스러운 idle 애니메이션을 스트리밍합니다.

        Args:
            session_id: 세션 ID
            duration: 스트리밍 지속 시간 (초). -1이면 무한

        Yields:
            VideoFrame: Idle 비디오 프레임
        """
        self._ensure_initialized()

        # 세션의 현재 감정 상태 적용
        if session_id:
            session = self.get_session(session_id)
            if session:
                self.renderer.set_emotion(session.current_emotion)

        async for frame in self.renderer.render_idle_stream(duration=duration):
            yield frame

    def _update_metrics(self, latency_ms: float) -> None:
        """성능 메트릭 업데이트"""
        self._metrics["total_requests"] += 1
        self._metrics["last_latency_ms"] = latency_ms

        # 이동 평균 계산
        n = self._metrics["total_requests"]
        avg = self._metrics["avg_latency_ms"]
        self._metrics["avg_latency_ms"] = avg + (latency_ms - avg) / n

    def get_metrics(self) -> Dict[str, Any]:
        """현재 메트릭 반환"""
        return self._metrics.copy()

    async def cleanup(self) -> None:
        """모든 리소스 정리"""
        logger.info("Cleaning up Pipeline Orchestrator...")

        await asyncio.gather(
            self.stt.cleanup(),
            self.llm.cleanup(),
            self.tts.cleanup(),
            self.renderer.cleanup(),
        )

        self._sessions.clear()
        self._initialized = False
        logger.info("Pipeline Orchestrator cleaned up")
