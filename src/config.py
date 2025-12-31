"""
Configuration management for the Realtime AI Avatar Service.

환경 변수 기반 설정 관리 모듈
"""

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Device Configuration
    device: str = Field(default="cuda", description="Compute device (cuda, cpu, mps)")

    # LLM Configuration
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_provider: Literal["openai"] = Field(
        default="openai", description="LLM provider"
    )
    llm_model: str = Field(
        default="gpt-4o", description="LLM model name"
    )

    # LiveKit Configuration
    livekit_api_key: str = Field(default="", description="LiveKit API key")
    livekit_api_secret: str = Field(default="", description="LiveKit API secret")
    livekit_url: str = Field(
        default="wss://localhost:7880", description="LiveKit server URL"
    )

    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )

    # Paths
    voice_sample_path: str = Field(
        default="assets/voice_sample.wav", description="Path to voice sample for TTS"
    )
    idle_loops_dir: str = Field(
        default="assets/idle_loops", description="Directory for idle loop videos"
    )
    avatars_dir: str = Field(
        default="assets/avatars", description="Directory for avatar images"
    )
    default_avatar_image: str = Field(
        default="assets/avatars/avata_image.png", description="Default avatar image path"
    )
    driving_video_path: Optional[str] = Field(
        default=None, description="Driving video path for idle animation (LivePortrait)"
    )
    voices_dir: str = Field(
        default="assets/voices", description="Directory for Zonos TTS voice profiles"
    )

    # Avatar Settings
    system_prompt: str = Field(
        default="당신은 친절하고 공감능력이 뛰어난 AI 어시스턴트입니다. 사용자의 감정에 맞춰 대화해주세요.",
        description="System prompt for LLM",
    )
    default_language: Literal["ko", "en", "zh", "ja"] = Field(
        default="ko", description="Default language"
    )

    # Performance Settings
    target_fps: int = Field(default=30, description="Target frame rate")
    max_latency_ms: int = Field(
        default=800, description="Maximum acceptable latency in milliseconds"
    )
    audio_sample_rate: int = Field(
        default=16000, description="Audio sample rate for STT"
    )
    video_width: int = Field(default=784, description="Output video width (default: avata_ani.mp4 width)")
    video_height: int = Field(default=1176, description="Output video height (default: avata_ani.mp4 height)")

    # Lip Sync Settings
    fast_lipsync: bool = Field(
        default=False,
        description="Use fast lip sync simulation instead of MuseTalk (faster but lower quality)"
    )
    musetalk_use_tensorrt: bool = Field(
        default=False,
        description="Use TensorRT for MuseTalk inference (faster but requires TensorRT installation and engine compilation)"
    )
    musetalk_tensorrt_workspace_size: int = Field(
        default=1024 * 1024 * 1024,  # 1GB
        description="TensorRT workspace size in bytes (for engine optimization)"
    )
    musetalk_tensorrt_fp16: bool = Field(
        default=True,
        description="Use FP16 precision for TensorRT engines (faster, may slightly reduce quality)"
    )

    # Idle Loop Settings
    idle_loop_cache_enabled: bool = Field(
        default=True,
        description="Enable frame caching for idle loop videos (faster but uses more memory)"
    )
    idle_loop_cache_max_frames: int = Field(
        default=300,
        description="Maximum number of frames to cache (300 frames = ~10 seconds @ 30fps)"
    )
    idle_loop_cache_max_duration_seconds: float = Field(
        default=10.0,
        description="Maximum video duration to cache in seconds (videos longer than this will stream from file)"
    )
    disable_server_idle_stream: bool = Field(
        default=True,
        description="Disable server-side idle stream (use local video playback in frontend instead, saves bandwidth and server resources). Default: True (local video playback enabled)"
    )

    # TTS Settings
    tts_provider: Literal["edge", "zonos", "elevenlabs"] = Field(
        default="edge", description="TTS provider (edge: fast/free, zonos: voice cloning/slow, elevenlabs: API/paid)"
    )
    tts_voice: str = Field(
        default="default", description="TTS voice ID (Zonos voice profile ID or ElevenLabs voice ID)"
    )
    # Edge TTS Settings
    edge_voice: str = Field(
        default="ko-KR-SunHiNeural", description="Edge TTS voice (ko-KR-SunHiNeural, ko-KR-InJoonNeural, etc.)"
    )
    # ElevenLabs Settings
    elevenlabs_api_key: str = Field(
        default="", description="ElevenLabs API key"
    )
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM", description="ElevenLabs voice ID (default: Rachel)"
    )
    elevenlabs_model_id: str = Field(
        default="eleven_multilingual_v2", description="ElevenLabs model ID (multilingual v2 supports Korean)"
    )
    tts_sample_rate: int = Field(default=24000, description="TTS audio sample rate")
    tts_chunk_size: int = Field(
        default=4096, description="TTS audio chunk size in bytes"
    )
    tts_initial_chunk_size: int = Field(
        default=1024, description="Initial TTS chunk size for faster first output"
    )
    tts_min_text_for_early_synthesis: int = Field(
        default=20, description="Minimum text length to trigger early synthesis"
    )
    tts_stream_buffer_ms: int = Field(
        default=100, description="Stream buffer size in milliseconds"
    )
    tts_enable_compile: bool = Field(
        default=False,
        description="Enable torch.compile() for Zonos TTS (Linux only, faster but may cause issues)"
    )

    # STT Model Settings
    stt_model_path: str = Field(
        default="iic/SenseVoiceSmall",
        description="STT model path (ModelScope repository path)"
    )

    # Session Settings
    session_ttl_seconds: int = Field(
        default=1800, description="Session TTL in seconds (default: 30 minutes)"
    )
    session_cleanup_interval_seconds: int = Field(
        default=60, description="Session cleanup check interval in seconds"
    )

    # Audio Buffer Settings
    audio_buffer_max_seconds: int = Field(
        default=10, description="Maximum audio buffer duration in seconds"
    )

    # Security Settings
    max_message_size_bytes: int = Field(
        default=1048576, description="Maximum WebSocket message size (1MB)"
    )
    max_text_length: int = Field(
        default=10000, description="Maximum chat text length (characters)"
    )
    max_audio_chunk_size: int = Field(
        default=1048576, description="Maximum audio chunk size (1MB)"
    )
    max_system_prompt_length: int = Field(
        default=2000, description="Maximum custom system prompt length"
    )
    allow_custom_system_prompt: bool = Field(
        default=False, description="Allow users to set custom system prompts"
    )

    # WebSocket Reconnection Settings
    # TTS 처리가 오래 걸릴 수 있으므로 (50초 이상) 타임아웃을 늘림
    websocket_heartbeat_interval: int = Field(
        default=120, description="Heartbeat ping interval in seconds"
    )
    websocket_heartbeat_timeout: int = Field(
        default=300, description="Heartbeat response timeout in seconds (TTS 처리 시간 고려)"
    )
    websocket_reconnect_window: int = Field(
        default=120, description="Time window for session reconnection in seconds"
    )

    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    @property
    def llm_api_key(self) -> str:
        """Get the API key for the configured LLM provider."""
        return self.openai_api_key

    def get_device(self) -> str:
        """Get the compute device, with fallback to CPU if GPU is not available."""
        try:
            import torch

            if self.device == "cuda":
                if torch.cuda.is_available():
                    return "cuda"
                print("CUDA not available, falling back to CPU")
                return "cpu"
            elif self.device == "mps":
                if torch.backends.mps.is_available():
                    return "mps"
                print("MPS not available, falling back to CPU")
                return "cpu"
            return self.device
        except ImportError:
            return "cpu"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
