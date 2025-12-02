"""
Voice Management API Routes.

음성 프로필 관리 및 Zonos TTS 제어 API
"""

import logging
import uuid
from typing import Optional, List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voices", tags=["Voice Management"])

# 전역 참조 (main.py에서 설정)
_zonos_tts = None


def set_zonos_tts(zonos_tts):
    """Zonos TTS 인스턴스 설정"""
    global _zonos_tts
    _zonos_tts = zonos_tts


def get_zonos_tts():
    """Zonos TTS 의존성"""
    if _zonos_tts is None:
        raise HTTPException(status_code=503, detail="Zonos TTS not initialized")
    return _zonos_tts


# ============ Pydantic Models ============

class VoiceProfileResponse(BaseModel):
    """음성 프로필 응답"""
    id: str
    name: str
    description: str
    language: str
    duration_seconds: float
    sample_rate: int
    created_at: str
    updated_at: str


class VoiceProfileCreate(BaseModel):
    """음성 프로필 생성 요청 (JSON 부분)"""
    name: str
    description: str = ""
    language: str = "en"


class VoiceProfileUpdate(BaseModel):
    """음성 프로필 업데이트 요청"""
    name: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None


class SynthesizeRequest(BaseModel):
    """TTS 합성 요청"""
    text: str
    voice_id: Optional[str] = None
    language: str = "en"
    emotion: str = "neutral"
    speaking_rate: float = 1.0


class VoiceListResponse(BaseModel):
    """음성 목록 응답"""
    voices: List[VoiceProfileResponse]
    total: int
    supported_languages: List[str]


class LanguageInfo(BaseModel):
    """언어 정보"""
    code: str
    name: str
    supported: bool


# ============ API Endpoints ============

@router.get("", response_model=VoiceListResponse)
async def list_voices():
    """
    모든 음성 프로필 목록 조회

    Returns:
        VoiceListResponse: 음성 프로필 목록
    """
    try:
        from ..models.integrations.zonos_tts import ZonosTTSModel

        zonos = get_zonos_tts()
        profiles = zonos.list_voice_profiles()

        return VoiceListResponse(
            voices=[
                VoiceProfileResponse(
                    id=p.id,
                    name=p.name,
                    description=p.description,
                    language=p.language,
                    duration_seconds=p.duration_seconds,
                    sample_rate=p.sample_rate,
                    created_at=p.created_at,
                    updated_at=p.updated_at,
                )
                for p in profiles
            ],
            total=len(profiles),
            supported_languages=ZonosTTSModel.SUPPORTED_LANGUAGES,
        )
    except HTTPException:
        # Zonos TTS가 초기화되지 않은 경우 빈 목록 반환
        return VoiceListResponse(
            voices=[],
            total=0,
            supported_languages=["en", "ja", "zh", "fr", "de"],
        )


@router.post("", response_model=VoiceProfileResponse)
async def create_voice(
    audio: UploadFile = File(..., description="참조 오디오 파일 (WAV/MP3)"),
    name: str = Form(..., description="음성 이름"),
    description: str = Form("", description="음성 설명"),
    language: str = Form("en", description="언어 코드 (en, ja, zh, fr, de)"),
):
    """
    새 음성 프로필 생성 (음성 복제)

    - 10-30초 길이의 깨끗한 음성 파일 권장
    - 지원 포맷: WAV, MP3, M4A, OGG
    - 최대 파일 크기: 50MB

    Returns:
        VoiceProfileResponse: 생성된 음성 프로필
    """
    zonos = get_zonos_tts()

    # 파일 크기 검증
    content = await audio.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    if len(content) < 1000:
        raise HTTPException(status_code=400, detail="File too small")

    # 고유 ID 생성
    voice_id = str(uuid.uuid4())[:8]

    # 음성 프로필 생성
    profile = await zonos.create_voice_profile(
        voice_id=voice_id,
        name=name,
        audio_data=content,
        description=description,
        language=language,
    )

    if profile is None:
        raise HTTPException(status_code=500, detail="Failed to create voice profile")

    return VoiceProfileResponse(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        language=profile.language,
        duration_seconds=profile.duration_seconds,
        sample_rate=profile.sample_rate,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


@router.get("/{voice_id}", response_model=VoiceProfileResponse)
async def get_voice(voice_id: str):
    """
    음성 프로필 상세 조회

    Args:
        voice_id: 음성 프로필 ID

    Returns:
        VoiceProfileResponse: 음성 프로필 정보
    """
    zonos = get_zonos_tts()
    profile = zonos.get_voice_profile(voice_id)

    if profile is None:
        raise HTTPException(status_code=404, detail="Voice profile not found")

    return VoiceProfileResponse(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        language=profile.language,
        duration_seconds=profile.duration_seconds,
        sample_rate=profile.sample_rate,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


@router.put("/{voice_id}", response_model=VoiceProfileResponse)
async def update_voice(voice_id: str, request: VoiceProfileUpdate):
    """
    음성 프로필 업데이트

    Args:
        voice_id: 음성 프로필 ID
        request: 업데이트 내용

    Returns:
        VoiceProfileResponse: 업데이트된 음성 프로필
    """
    zonos = get_zonos_tts()

    profile = await zonos.update_voice_profile(
        voice_id=voice_id,
        name=request.name,
        description=request.description,
        language=request.language,
    )

    if profile is None:
        raise HTTPException(status_code=404, detail="Voice profile not found")

    return VoiceProfileResponse(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        language=profile.language,
        duration_seconds=profile.duration_seconds,
        sample_rate=profile.sample_rate,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


@router.delete("/{voice_id}")
async def delete_voice(voice_id: str):
    """
    음성 프로필 삭제

    Args:
        voice_id: 음성 프로필 ID

    Returns:
        삭제 결과
    """
    zonos = get_zonos_tts()

    success = await zonos.delete_voice_profile(voice_id)

    if not success:
        raise HTTPException(status_code=404, detail="Voice profile not found")

    return {"status": "deleted", "voice_id": voice_id}


@router.get("/{voice_id}/audio")
async def get_voice_audio(voice_id: str):
    """
    음성 프로필 원본 오디오 다운로드

    Args:
        voice_id: 음성 프로필 ID

    Returns:
        오디오 파일 (WAV)
    """
    from fastapi.responses import Response

    zonos = get_zonos_tts()

    audio_data = await zonos.get_voice_audio(voice_id)

    if audio_data is None:
        raise HTTPException(status_code=404, detail="Voice audio not found")

    return Response(
        content=audio_data,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"attachment; filename={voice_id}.wav"
        },
    )


@router.post("/{voice_id}/preview")
async def preview_voice(voice_id: str, request: SynthesizeRequest):
    """
    음성 미리듣기 (TTS 합성)

    Args:
        voice_id: 음성 프로필 ID
        request: 합성 요청

    Returns:
        합성된 오디오 (WAV)
    """
    from fastapi.responses import Response
    import io
    import wave

    zonos = get_zonos_tts()

    # 음성 프로필 확인
    profile = zonos.get_voice_profile(voice_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Voice profile not found")

    # TTS 합성
    audio = await zonos.synthesize(
        text=request.text,
        voice_id=voice_id,
        language=request.language,
        emotion=request.emotion,
        speaking_rate=request.speaking_rate,
    )

    if len(audio) == 0:
        raise HTTPException(status_code=500, detail="Failed to synthesize audio")

    # WAV 형식으로 변환
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(zonos.sample_rate)
        wav_file.writeframes((audio * 32767).astype("int16").tobytes())

    wav_buffer.seek(0)

    return Response(
        content=wav_buffer.read(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": f"inline; filename=preview_{voice_id}.wav"
        },
    )


@router.post("/synthesize")
async def synthesize_text(request: SynthesizeRequest):
    """
    텍스트를 음성으로 변환

    Args:
        request: 합성 요청

    Returns:
        합성된 오디오 (WAV)
    """
    from fastapi.responses import Response
    import io
    import wave

    zonos = get_zonos_tts()

    # TTS 합성
    audio = await zonos.synthesize(
        text=request.text,
        voice_id=request.voice_id,
        language=request.language,
        emotion=request.emotion,
        speaking_rate=request.speaking_rate,
    )

    if len(audio) == 0:
        raise HTTPException(status_code=500, detail="Failed to synthesize audio")

    # WAV 형식으로 변환
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(zonos.sample_rate)
        wav_file.writeframes((audio * 32767).astype("int16").tobytes())

    wav_buffer.seek(0)

    return Response(
        content=wav_buffer.read(),
        media_type="audio/wav",
    )


@router.get("/languages/supported")
async def get_supported_languages():
    """
    지원되는 언어 목록

    Returns:
        언어 목록
    """
    return {
        "languages": [
            {"code": "en", "name": "English", "supported": True},
            {"code": "ja", "name": "Japanese", "supported": True},
            {"code": "zh", "name": "Chinese", "supported": True},
            {"code": "fr", "name": "French", "supported": True},
            {"code": "de", "name": "German", "supported": True},
            {"code": "ko", "name": "Korean", "supported": False},
        ]
    }


@router.get("/emotions/supported")
async def get_supported_emotions():
    """
    지원되는 감정 목록

    Returns:
        감정 목록
    """
    from ..models.integrations.zonos_tts import ZonosTTSModel

    return {
        "emotions": list(ZonosTTSModel.EMOTIONS.keys())
    }
