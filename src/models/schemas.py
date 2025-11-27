"""
Pydantic schemas for API and internal data structures.

API 요청/응답 및 내부 데이터 구조를 위한 Pydantic 모델
"""

from datetime import datetime
from typing import Optional, List, Any
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .emotion import Emotion


class PipelineState(str, Enum):
    """파이프라인 상태"""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


class AudioChunk(BaseModel):
    """오디오 청크 데이터"""

    data: bytes = Field(..., description="Raw audio data")
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="Timestamp of the chunk",
    )


class STTResult(BaseModel):
    """STT 결과"""

    text: str = Field(default="", description="Transcribed text")
    emotion: Emotion = Field(default=Emotion.NEUTRAL, description="Detected emotion")
    language: str = Field(default="unknown", description="Detected language code")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score"
    )
    is_final: bool = Field(default=True, description="Whether this is a final result")
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )


class LLMResponse(BaseModel):
    """LLM 응답"""

    text: str = Field(..., description="Generated text")
    is_complete: bool = Field(
        default=False, description="Whether this is the complete response"
    )
    token_count: int = Field(default=0, description="Number of tokens generated")


class TTSChunk(BaseModel):
    """TTS 오디오 청크"""

    audio_data: bytes = Field(..., description="Audio data bytes")
    sample_rate: int = Field(default=24000, description="Sample rate in Hz")
    duration_ms: float = Field(default=0.0, description="Duration in milliseconds")
    is_last: bool = Field(
        default=False, description="Whether this is the last chunk"
    )


class VideoFrame(BaseModel):
    """비디오 프레임"""

    data: bytes = Field(..., description="Encoded frame data (JPEG/H.264)")
    width: int = Field(default=512, description="Frame width")
    height: int = Field(default=512, description="Frame height")
    timestamp: float = Field(..., description="Frame timestamp")
    frame_index: int = Field(default=0, description="Frame index in sequence")
    encoding: str = Field(default="jpeg", description="Encoding format")


class AvatarSession(BaseModel):
    """아바타 세션 정보"""

    session_id: UUID = Field(default_factory=uuid4, description="Unique session ID")
    created_at: datetime = Field(
        default_factory=datetime.now, description="Session creation time"
    )
    avatar_id: str = Field(default="default", description="Avatar identifier")
    current_emotion: Emotion = Field(
        default=Emotion.NEUTRAL, description="Current avatar emotion"
    )
    pipeline_state: PipelineState = Field(
        default=PipelineState.IDLE, description="Current pipeline state"
    )
    conversation_history: List[dict] = Field(
        default_factory=list, description="Conversation history"
    )
    total_interactions: int = Field(default=0, description="Total number of interactions")
    last_activity: datetime = Field(
        default_factory=datetime.now, description="Last activity timestamp"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


# API Request/Response Models


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(..., description="Service status")
    pipeline_ready: bool = Field(..., description="Whether pipeline is ready")
    version: str = Field(default="1.0.0", description="Service version")
    uptime_seconds: float = Field(default=0.0, description="Service uptime in seconds")
    gpu_available: bool = Field(default=False, description="Whether GPU is available")


class TokenRequest(BaseModel):
    """LiveKit token request"""

    room_name: str = Field(..., description="Room name to join")
    participant_name: str = Field(..., description="Participant display name")
    participant_identity: Optional[str] = Field(
        None, description="Unique participant identity"
    )


class TokenResponse(BaseModel):
    """LiveKit token response"""

    token: str = Field(..., description="JWT token for LiveKit")
    room_name: str = Field(..., description="Room name")
    livekit_url: str = Field(..., description="LiveKit server URL")


class SessionCreateRequest(BaseModel):
    """Avatar session creation request"""

    avatar_id: str = Field(default="default", description="Avatar to use")
    system_prompt: Optional[str] = Field(
        None, description="Custom system prompt for this session"
    )
    voice_id: Optional[str] = Field(None, description="Custom voice ID for TTS")
    language: str = Field(default="ko", description="Primary language")


class SessionCreateResponse(BaseModel):
    """Avatar session creation response"""

    session_id: str = Field(..., description="Created session ID")
    avatar_id: str = Field(..., description="Avatar being used")
    websocket_url: str = Field(..., description="WebSocket URL for this session")
    livekit_token: Optional[str] = Field(
        None, description="LiveKit token if using WebRTC"
    )


class WebSocketMessage(BaseModel):
    """WebSocket 메시지 구조"""

    type: str = Field(..., description="Message type")
    payload: Any = Field(default=None, description="Message payload")
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="Message timestamp",
    )


class ErrorResponse(BaseModel):
    """에러 응답"""

    error: str = Field(..., description="Error message")
    error_code: str = Field(default="UNKNOWN_ERROR", description="Error code")
    details: Optional[dict] = Field(None, description="Additional error details")
