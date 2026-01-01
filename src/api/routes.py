"""
API Routes for the Realtime AI Avatar Service.

REST API 엔드포인트 정의
"""

import logging
import time
import io
from typing import Optional
from uuid import UUID

import numpy as np

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form

from ..config import Settings, get_settings
from ..models.schemas import (
    HealthResponse,
    TokenRequest,
    TokenResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# 전역 참조 (main.py에서 설정)
_pipeline = None
_livekit = None
_start_time = time.time()
_initialization_complete = False  # 초기화 완료 플래그


def set_services(pipeline, livekit):
    """서비스 인스턴스 설정 (main.py에서 호출)"""
    global _pipeline, _livekit
    _pipeline = pipeline
    _livekit = livekit

def set_initialization_complete(complete: bool = True):
    """초기화 완료 플래그 설정 (main.py에서 호출)"""
    global _initialization_complete
    _initialization_complete = complete

def get_initialization_complete() -> bool:
    """초기화 완료 여부 확인"""
    return _initialization_complete


def get_pipeline():
    """파이프라인 의존성"""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return _pipeline


def get_livekit():
    """LiveKit 의존성"""
    if _livekit is None:
        raise HTTPException(status_code=503, detail="LiveKit service not initialized")
    return _livekit


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check(settings: Settings = Depends(get_settings)):
    """
    서버 상태 확인

    Returns:
        HealthResponse: 서버 상태 정보
    """
    # 🔑 초기화 완료 여부 확인
    from ..main import _initialization_complete
    if not _initialization_complete:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service is still initializing")
    # GPU 가용성 확인 (CUDA 또는 MPS)
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available() or torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        pass

    return HealthResponse(
        status="healthy",
        pipeline_ready=_pipeline is not None,
        version="1.0.0",
        uptime_seconds=time.time() - _start_time,
        gpu_available=gpu_available,
    )


@router.post(
    "/api/generate-token",
    response_model=TokenResponse,
    tags=["LiveKit"],
    responses={503: {"model": ErrorResponse}},
)
async def generate_livekit_token(
    request: TokenRequest,
    settings: Settings = Depends(get_settings),
    livekit=Depends(get_livekit),
):
    """
    LiveKit 룸 접속 토큰 생성

    Args:
        request: 토큰 요청 정보

    Returns:
        TokenResponse: 생성된 토큰
    """
    token = livekit.create_token(
        room_name=request.room_name,
        participant_name=request.participant_name,
        participant_identity=request.participant_identity,
    )

    if not token:
        raise HTTPException(
            status_code=503,
            detail="Failed to generate token. LiveKit may not be configured.",
        )

    return TokenResponse(
        token=token,
        room_name=request.room_name,
        livekit_url=settings.livekit_url,
    )


@router.post(
    "/api/avatar/create",
    response_model=SessionCreateResponse,
    tags=["Avatar"],
    responses={503: {"model": ErrorResponse}},
)
async def create_avatar_session(
    request: SessionCreateRequest,
    settings: Settings = Depends(get_settings),
    pipeline=Depends(get_pipeline),
    livekit=Depends(get_livekit),
):
    """
    새 아바타 세션 생성

    Args:
        request: 세션 생성 요청

    Returns:
        SessionCreateResponse: 생성된 세션 정보
    """
    # 시스템 프롬프트 검증
    system_prompt = settings.system_prompt
    if request.system_prompt:
        if not settings.allow_custom_system_prompt:
            logger.warning("Custom system prompt rejected: not allowed by configuration")
        elif len(request.system_prompt) > settings.max_system_prompt_length:
            logger.warning(f"Custom system prompt rejected: too long ({len(request.system_prompt)} > {settings.max_system_prompt_length})")
        else:
            system_prompt = request.system_prompt

    # 세션 생성
    session = pipeline.create_session(
        avatar_id=request.avatar_id,
        system_prompt=system_prompt,
    )

    # WebSocket URL 생성
    ws_url = f"ws://{settings.host}:{settings.port}/ws/avatar/{session.session_id}"

    # LiveKit 토큰 생성 (선택적)
    livekit_token = None
    if settings.livekit_api_key:
        room_name = f"avatar-{session.session_id}"
        livekit_token = livekit.create_token(
            room_name=room_name,
            participant_name="avatar",
            can_publish=True,
            can_subscribe=True,
        )

    return SessionCreateResponse(
        session_id=str(session.session_id),
        avatar_id=request.avatar_id,
        websocket_url=ws_url,
        livekit_token=livekit_token,
    )


@router.get(
    "/api/avatar/{session_id}",
    tags=["Avatar"],
    responses={404: {"model": ErrorResponse}},
)
async def get_avatar_session(
    session_id: str,
    pipeline=Depends(get_pipeline),
):
    """
    세션 정보 조회

    Args:
        session_id: 세션 ID

    Returns:
        세션 정보
    """
    try:
        uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    session = pipeline.get_session(uuid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": str(session.session_id),
        "avatar_id": session.avatar_id,
        "current_emotion": session.current_emotion.value,
        "pipeline_state": session.pipeline_state.value,
        "total_interactions": session.total_interactions,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
    }


@router.delete(
    "/api/avatar/{session_id}",
    tags=["Avatar"],
    responses={404: {"model": ErrorResponse}},
)
async def delete_avatar_session(
    session_id: str,
    pipeline=Depends(get_pipeline),
):
    """
    세션 삭제

    Args:
        session_id: 세션 ID

    Returns:
        삭제 결과
    """
    try:
        uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    success = pipeline.delete_session(uuid)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "deleted", "session_id": session_id}


@router.get("/api/metrics", tags=["Metrics"])
async def get_metrics(pipeline=Depends(get_pipeline)):
    """
    파이프라인 성능 메트릭 조회

    Returns:
        성능 메트릭
    """
    return pipeline.get_metrics()


@router.get("/api/avatars", tags=["Avatar"])
async def list_available_avatars():
    """
    사용 가능한 아바타 목록

    Returns:
        아바타 목록
    """
    # TODO: 실제 아바타 목록 구현
    return {
        "avatars": [
            {
                "id": "default",
                "name": "Default Avatar",
                "description": "기본 아바타",
            }
        ]
    }


@router.get("/api/emotions", tags=["Avatar"])
async def list_emotions():
    """
    지원되는 감정 목록

    Returns:
        감정 목록
    """
    from ..models.emotion import Emotion

    return {
        "emotions": [e.value for e in Emotion],
    }


@router.get("/api/avatar/{session_id}/emotion", tags=["Avatar"])
async def get_session_emotion(
    session_id: str,
    pipeline=Depends(get_pipeline),
):
    """
    세션의 현재 감정 분석 결과 조회

    Args:
        session_id: 세션 ID

    Returns:
        감정 분석 결과
    """
    try:
        uuid = UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    session = pipeline.get_session(uuid)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": str(session.session_id),
        "current_emotion": session.current_emotion.value,
        "pipeline_state": session.pipeline_state.value,
    }


@router.post("/api/analyze-emotion", tags=["Avatar"])
async def analyze_emotion_from_audio(
    audio_file: UploadFile = File(..., description="음성 파일 (WAV/MP3 형식)"),
    sample_rate: int = Form(default=16000, description="샘플 레이트 (기본값: 16000)"),
    pipeline=Depends(get_pipeline),
):
    """
    업로드된 음성 파일에서 감정 분석

    Request Body (multipart/form-data):
        - audio_file: 음성 파일 (WAV/MP3 형식)
        - sample_rate: 샘플 레이트 (선택, 기본값: 16000)

    Returns:
        감정 분석 결과 (텍스트, 감정, 언어, 신뢰도)
    """
    try:
        import soundfile as sf
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="soundfile library is required for audio analysis. Please install it: pip install soundfile"
        )

    try:
        # 파일 읽기
        audio_bytes = await audio_file.read()
        
        # 오디오 디코딩
        try:
            # soundfile으로 오디오 로드
            audio_data, sr = sf.read(io.BytesIO(audio_bytes))
            
            # 모노로 변환 (스테레오인 경우)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # 샘플 레이트 변환 (필요한 경우)
            if sr != sample_rate:
                try:
                    from scipy import signal
                    num_samples = int(len(audio_data) * sample_rate / sr)
                    audio_data = signal.resample(audio_data, num_samples)
                except ImportError:
                    # scipy가 없으면 간단한 리샘플링 (품질 저하 가능)
                    logger.warning("scipy not available, using simple resampling")
                    import math
                    indices = np.linspace(0, len(audio_data) - 1, int(len(audio_data) * sample_rate / sr))
                    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)
            
            # 정규화
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
        except Exception as e:
            logger.error(f"Failed to decode audio file: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode audio file: {str(e)}"
            )

        # STT 모듈로 감정 분석
        stt_result = await pipeline.stt.transcribe(
            audio=audio_data,
            sample_rate=sample_rate,
            language="auto",
        )

        return {
            "text": stt_result.text,
            "emotion": stt_result.emotion.value,
            "language": stt_result.language,
            "confidence": stt_result.confidence,
            "processing_time_ms": stt_result.processing_time_ms,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze emotion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze emotion: {str(e)}"
        )