"""
Main FastAPI Application for Realtime AI Avatar Service.

실시간 AI 아바타 서비스 메인 엔트리포인트
"""

# 🔑 CODE VERSION MARKER - 이 버전이 출력되면 최신 코드가 실행 중인 것입니다
CODE_VERSION = "2026-01-02-v5-MOUTH-POSITION-FIX"

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
from .api.routes import router, set_services, set_initialization_complete
from .api.voice_routes import router as voice_router, set_zonos_tts
from .api.websocket import set_websocket_handler, get_websocket_handler
from .models.integrations.zonos_tts import ZonosTTSModel

# 🔑 출력 버퍼링 비활성화 (즉시 출력) - 반드시 로깅 설정 전에 실행
import os
os.environ['PYTHONUNBUFFERED'] = '1'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)


# 🔑 즉시 flush하는 StreamHandler
class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        FlushStreamHandler(sys.stdout),  # 🔑 즉시 flush하는 핸들러 사용
    ],
    force=True,  # 🔑 기존 핸들러를 강제로 교체
)

# 🔑 루트 로거 레벨 명시적 설정
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# 🔑 uvicorn 로거 레벨 설정 (INFO로 설정하여 모든 로그 출력)
# uvicorn이 자체 로깅 설정을 사용하지 않도록 핸들러 제거 후 재설정
uvicorn_logger = logging.getLogger("uvicorn")
uvicorn_logger.handlers = []  # 기존 핸들러 제거
uvicorn_logger.setLevel(logging.INFO)
uvicorn_logger.propagate = True  # 루트 로거로 전파

uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.handlers = []  # 기존 핸들러 제거
uvicorn_access_logger.setLevel(logging.INFO)
uvicorn_access_logger.propagate = True  # 루트 로거로 전파

uvicorn_error_logger = logging.getLogger("uvicorn.error")
uvicorn_error_logger.handlers = []  # 기존 핸들러 제거
uvicorn_error_logger.setLevel(logging.INFO)
uvicorn_error_logger.propagate = True  # 루트 로거로 전파

# 🔑 모든 주요 모듈의 로거 레벨 명시적 설정 (INFO 레벨 보장)
for module_name in ['src', 'src.api', 'src.api.websocket', 'src.pipeline', 'src.pipeline.orchestrator', 
                    'src.pipeline.avatar_renderer', 'src.pipeline.tts_module', 'src.pipeline.llm_module',
                    'src.models', 'src.models.integrations.musetalk', 'src.models.integrations.edge_tts']:
    module_logger = logging.getLogger(module_name)
    module_logger.setLevel(logging.INFO)
    # 상위 로거의 핸들러를 사용하도록 설정 (propagate=True가 기본값이지만 명시적으로 설정)
    module_logger.propagate = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 🔑 모든 로거에 즉시 flush 적용
for handler in logging.root.handlers:
    handler.flush()

# 전역 인스턴스
pipeline: Optional[PipelineOrchestrator] = None
livekit: Optional[LiveKitService] = None
tts_cache: Optional[TTSCacheService] = None
zonos_tts: Optional[ZonosTTSModel] = None
_initialization_complete: bool = False  # 초기화 완료 플래그


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 리소스 관리"""
    global pipeline, livekit, tts_cache, zonos_tts, _initialization_complete

    # 🔑 lifespan 시작 시 로깅 설정 재적용 (uvicorn이 로깅을 덮어쓴 경우 대비)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # 모든 주요 로거의 레벨 재설정
    for logger_name in ['uvicorn', 'uvicorn.access', 'uvicorn.error', 'src', 'src.api', 'src.api.websocket', 
                        'src.pipeline', 'src.pipeline.orchestrator', 'src.pipeline.avatar_renderer',
                        'src.pipeline.tts_module', 'src.pipeline.llm_module', 'src.models', 
                        'src.models.integrations.musetalk', 'src.models.integrations.edge_tts']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = True
        # uvicorn 로거의 경우 핸들러 제거하여 루트 로거로 전파
        if logger_name.startswith('uvicorn'):
            logger.handlers = []
    
    logger.info("=" * 60)
    logger.info("🔧 Logging configuration re-applied in lifespan")
    logger.info("=" * 60)

    settings = get_settings()

    # 🔑 Settings 디버그 로깅 (환경 변수 확인)
    import os
    logger.info("=" * 60)
    logger.info("🔧 Settings 디버그 정보:")
    width_env = os.environ.get('VIDEO_WIDTH', 'NOT SET')
    height_env = os.environ.get('VIDEO_HEIGHT', 'NOT SET')

    if settings.video_width is None and settings.video_height is None:
        logger.info("   📐 video_width/height: AUTO-DETECT (비디오에서 자동 감지)")
    else:
        logger.info(f"   video_width (from Settings): {settings.video_width}")
        logger.info(f"   video_height (from Settings): {settings.video_height}")

    logger.info(f"   VIDEO_WIDTH env var: {width_env}")
    logger.info(f"   VIDEO_HEIGHT env var: {height_env}")
    logger.info("=" * 60)

    logger.info("=" * 60)
    logger.info("Starting Realtime AI Avatar Service")
    logger.info(f"🔑 CODE VERSION: {CODE_VERSION}")
    print(f"\n{'='*60}", flush=True)
    print(f"🔑 CODE VERSION: {CODE_VERSION}", flush=True)
    print(f"{'='*60}\n", flush=True)
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
        
        # 🔑 초기화 완료 플래그 설정 (서버가 요청을 처리할 준비가 되었음을 표시)
        _initialization_complete = True
        set_initialization_complete(True)  # routes.py의 플래그도 설정
        logger.info("✅ 모든 서비스 초기화 완료 - 클라이언트 연결 허용")
        
        # 🔑 초기화 완료 후 로깅 설정 최종 확인 및 재적용
        # (uvicorn이 로깅을 덮어쓸 수 있으므로 마지막에 다시 확인)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # uvicorn 로거들 재설정
        for uvicorn_logger_name in ['uvicorn', 'uvicorn.access', 'uvicorn.error']:
            uvicorn_logger = logging.getLogger(uvicorn_logger_name)
            uvicorn_logger.handlers = []
            uvicorn_logger.setLevel(logging.INFO)
            uvicorn_logger.propagate = True
        # 모든 주요 모듈 로거 재설정
        for module_name in ['src', 'src.api', 'src.api.websocket']:
            module_logger = logging.getLogger(module_name)
            module_logger.setLevel(logging.INFO)
            module_logger.propagate = True
        
        logger.info("🔧 로깅 설정 최종 확인 완료 (요청 처리 준비 완료)")
        
        # 🔑 yield 이후에도 로깅이 유지되도록 한 번 더 확인
        # (실제로는 yield 이후에는 실행되지 않지만, 다음 요청 시를 대비)
        print("=" * 60, flush=True)
        print("[DEBUG] Initialization complete, logging should be working", flush=True)
        print("=" * 60, flush=True)

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
    # 🔑 테스트: 엔드포인트 진입 확인
    print("=" * 60, flush=True)
    print("[TEST] WebSocket 엔드포인트 호출됨!", flush=True)
    print("=" * 60, flush=True)
    logger.info("=" * 60)
    logger.info("[TEST] WebSocket 엔드포인트 호출됨!")
    logger.info("=" * 60)
    
    # 🔑 초기화 완료 확인
    if not _initialization_complete:
        logger.warning("⚠️ WebSocket 연결 시도: 서비스가 아직 초기화 중입니다")
        await websocket.close(code=1013, reason="Service initializing, please wait")
        return
    
    handler = get_websocket_handler()
    if handler is None:
        logger.warning("⚠️ WebSocket 연결 시도: 핸들러가 초기화되지 않았습니다")
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
    # 🔑 초기화 완료 확인
    if not _initialization_complete:
        logger.warning(f"⚠️ WebSocket 연결 시도 (session_id={session_id}): 서비스가 아직 초기화 중입니다")
        await websocket.close(code=1013, reason="Service initializing, please wait")
        return
    
    handler = get_websocket_handler()
    if handler is None:
        logger.warning(f"⚠️ WebSocket 연결 시도 (session_id={session_id}): 핸들러가 초기화되지 않았습니다")
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
        use_colors=True,  # 🔑 컬러 출력 활성화
        # 🔑 로그 설정을 명시적으로 전달
        log_config=None,  # 기본 로그 설정 사용하지 않음 (우리의 logging.basicConfig 사용)
    )


if __name__ == "__main__":
    run()
