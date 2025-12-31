"""
Main FastAPI Application for Realtime AI Avatar Service.

실시간 AI 아바타 서비스 메인 엔트리포인트
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import get_settings, Settings
from .pipeline.orchestrator import PipelineOrchestrator
from .services.livekit_service import LiveKitService
from .services.tts_cache_service import TTSCacheService
from .api.routes import router, set_services
from .api.voice_routes import router as voice_router, set_zonos_tts
from .api.websocket import set_websocket_handler, get_websocket_handler
from .models.integrations.zonos_tts import ZonosTTSModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# 전역 인스턴스
pipeline: Optional[PipelineOrchestrator] = None
livekit: Optional[LiveKitService] = None
tts_cache: Optional[TTSCacheService] = None
zonos_tts: Optional[ZonosTTSModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 리소스 관리"""
    global pipeline, livekit, tts_cache, zonos_tts

    settings = get_settings()

    # 🔑 Settings 디버그 로깅 (환경 변수 확인)
    import os
    logger.info("=" * 60)
    logger.info("🔧 Settings 디버그 정보:")
    logger.info(f"   video_width (from Settings): {settings.video_width}")
    logger.info(f"   video_height (from Settings): {settings.video_height}")
    logger.info(f"   VIDEO_WIDTH env var: {os.environ.get('VIDEO_WIDTH', 'NOT SET')}")
    logger.info(f"   VIDEO_HEIGHT env var: {os.environ.get('VIDEO_HEIGHT', 'NOT SET')}")
    logger.info("=" * 60)

    logger.info("=" * 60)
    logger.info("Starting Realtime AI Avatar Service")
    logger.info("=" * 60)

    try:
        # 파이프라인 초기화
        logger.info("Initializing pipeline...")
        pipeline = PipelineOrchestrator(settings)
        await pipeline.initialize()

        # LiveKit 서비스 초기화
        logger.info("Initializing LiveKit service...")
        livekit = LiveKitService(
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            url=settings.livekit_url,
        )
        await livekit.initialize()

        # TTS 캐시 서비스 초기화
        logger.info("Initializing TTS cache service...")
        tts_cache = TTSCacheService(
            redis_url=settings.redis_url if settings.redis_url else None,
        )
        await tts_cache.initialize()

        # Zonos TTS 초기화
        logger.info("Initializing Zonos TTS...")
        zonos_tts = ZonosTTSModel(
            device=settings.get_device(),
            voices_dir=settings.voices_dir,
        )
        await zonos_tts.initialize()

        # 서비스 등록
        set_services(pipeline, livekit)
        set_websocket_handler(pipeline)
        set_zonos_tts(zonos_tts)

        logger.info("=" * 60)
        logger.info("Avatar Pipeline initialized successfully!")
        logger.info(f"Server running at http://{settings.host}:{settings.port}")
        logger.info(f"WebSocket endpoint: ws://{settings.host}:{settings.port}/ws/avatar")
        logger.info("=" * 60)

        yield

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    finally:
        # 정리
        logger.info("Shutting down services...")

        if pipeline:
            await pipeline.cleanup()
        if livekit:
            await livekit.cleanup()
        if tts_cache:
            await tts_cache.cleanup()
        if zonos_tts:
            await zonos_tts.cleanup()

        logger.info("Shutdown complete")


# FastAPI 앱 생성
app = FastAPI(
    title="Realtime AI Avatar API",
    description="실시간 대화형 AI 아바타 서비스",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정
# 개발 환경에서는 모든 origin 허용 (WebSocket 연결을 위해 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경: 모든 origin 허용
    allow_credentials=False,  # allow_origins가 "*"일 때는 False여야 함
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 라우터 등록
app.include_router(router)
app.include_router(voice_router)


@app.get("/", tags=["Root"])
async def root():
    """
    루트 엔드포인트 - 서비스 정보 반환
    """
    return {
        "service": "Realtime AI Avatar Service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "websocket": "/ws/avatar",
        "endpoints": {
            "health_check": "GET /health",
            "create_session": "POST /api/avatar/create",
            "generate_token": "POST /api/generate-token",
            "list_avatars": "GET /api/avatars",
            "list_emotions": "GET /api/emotions",
            "get_session_emotion": "GET /api/avatar/{session_id}/emotion",
            "analyze_emotion": "POST /api/analyze-emotion",
            "metrics": "GET /api/metrics",
            "voices": "GET /api/voices",
            "voice_management": "POST /api/voices",
        }
    }


@app.websocket("/ws/avatar")
async def avatar_websocket_endpoint(websocket: WebSocket):
    """
    기본 아바타 WebSocket 엔드포인트

    Protocol:
    - Client → Server: 오디오 청크 (bytes) 또는 제어 메시지 (JSON)
    - Server → Client: 비디오 프레임 (bytes) 또는 상태 메시지 (JSON)
    """
    handler = get_websocket_handler()
    if handler is None:
        await websocket.close(code=1011, reason="Service not initialized")
        return

    await handler.handle_connection(websocket)


@app.websocket("/ws/avatar/{session_id}")
async def avatar_session_websocket_endpoint(
    websocket: WebSocket,
    session_id: str,
):
    """
    세션 기반 아바타 WebSocket 엔드포인트

    Args:
        session_id: 아바타 세션 ID
    """
    handler = get_websocket_handler()
    if handler is None:
        await websocket.close(code=1011, reason="Service not initialized")
        return

    await handler.handle_connection(websocket, session_id)


def run():
    """서버 실행 (CLI 엔트리포인트)"""
    settings = get_settings()

    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    run()
