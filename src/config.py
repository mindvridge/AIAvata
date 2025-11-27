"""
Configuration management for the Realtime AI Avatar Service.

환경 변수 기반 설정 관리 모듈
"""

from functools import lru_cache
from typing import Literal

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
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_provider: Literal["anthropic", "openai"] = Field(
        default="anthropic", description="LLM provider"
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514", description="LLM model name"
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
    video_width: int = Field(default=512, description="Output video width")
    video_height: int = Field(default=512, description="Output video height")

    # TTS Settings
    tts_sample_rate: int = Field(default=24000, description="TTS audio sample rate")
    tts_chunk_size: int = Field(
        default=4096, description="TTS audio chunk size in bytes"
    )

    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    @property
    def llm_api_key(self) -> str:
        """Get the API key for the configured LLM provider."""
        if self.llm_provider == "anthropic":
            return self.anthropic_api_key
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
