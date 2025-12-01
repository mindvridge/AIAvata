#!/usr/bin/env python3
"""
환경 체크 스크립트
모든 핵심 모듈과 모델 파일을 검증합니다.

Usage:
    python tools/check_env.py
"""

import sys
import asyncio
from pathlib import Path


def print_header(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}\n")


def check_mark(condition: bool) -> str:
    return "✅" if condition else "❌"


def check_python_modules():
    """Python 모듈 체크"""
    print_header("Python 모듈 체크")

    modules = {
        "torch": "PyTorch",
        "torchaudio": "TorchAudio",
        "funasr": "FunASR (STT)",
        "chatterbox": "Chatterbox (TTS)",
        "openai": "OpenAI SDK",
        "anthropic": "Anthropic SDK",
        "livekit": "LiveKit SDK",
        "diffusers": "Diffusers (VAE)",
        "transformers": "Transformers",
        "mediapipe": "MediaPipe",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
    }

    results = {}
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"  {check_mark(True)} {name}")
            results[module] = True
        except ImportError:
            print(f"  {check_mark(False)} {name} - 설치 필요")
            results[module] = False

    return results


def check_model_files():
    """모델 파일 체크"""
    print_header("모델 파일 체크")

    models = {
        "MuseTalk UNet": Path("models/musetalk/musetalkV15/unet.pth"),
        "MuseTalk VAE": Path("models/musetalk/sd-vae-ft-mse/config.json"),
    }

    results = {}
    for name, path in models.items():
        exists = path.exists()
        print(f"  {check_mark(exists)} {name}")
        if not exists:
            print(f"      경로: {path}")
        results[name] = exists

    return results


def check_env_file():
    """환경 변수 파일 체크"""
    print_header(".env 파일 체크")

    env_path = Path(".env")

    if not env_path.exists():
        print(f"  {check_mark(False)} .env 파일 없음")
        print("      → cp .env.example .env 실행 필요")
        return False

    # API 키 체크
    with open(env_path, "r") as f:
        content = f.read()

    keys = {
        "OPENAI_API_KEY": "your_openai" not in content and "OPENAI_API_KEY=" in content,
        "ANTHROPIC_API_KEY": "your_anthropic" not in content and "ANTHROPIC_API_KEY=" in content,
    }

    all_set = True
    for key, is_set in keys.items():
        if "your_" in content and key in content:
            print(f"  {check_mark(False)} {key} - 설정 필요")
            all_set = False
        elif key + "=" in content:
            print(f"  {check_mark(True)} {key}")
        else:
            print(f"  {check_mark(False)} {key} - 없음")
            all_set = False

    return all_set


def check_directories():
    """필수 디렉토리 체크"""
    print_header("디렉토리 체크")

    dirs = {
        "assets/avatars": Path("assets/avatars"),
        "assets/idle_loops": Path("assets/idle_loops"),
        "models/musetalk": Path("models/musetalk"),
        "frontend": Path("frontend"),
    }

    for name, path in dirs.items():
        exists = path.exists()
        has_files = exists and any(path.iterdir()) if exists else False
        if exists and has_files:
            print(f"  {check_mark(True)} {name}")
        elif exists:
            print(f"  ⚠️  {name} (비어있음)")
        else:
            print(f"  {check_mark(False)} {name}")


async def check_core_modules():
    """핵심 모듈 초기화 테스트"""
    print_header("핵심 모듈 테스트")

    # STT
    print("  STT (SenseVoice)...", end=" ", flush=True)
    try:
        from src.pipeline.stt_module import STTModule
        stt = STTModule(device="cpu")
        await stt.initialize()
        if stt.model:
            print("✅ SenseVoice")
        elif stt._use_whisper:
            print("⚠️  Whisper (fallback)")
        else:
            print("❌ 실패")
        await stt.cleanup()
    except Exception as e:
        print(f"❌ {e}")

    # TTS
    print("  TTS (Chatterbox)...", end=" ", flush=True)
    try:
        from src.models.integrations.chatterbox_tts import ChatterboxTTSModel
        tts = ChatterboxTTSModel(device="cpu")
        result = await tts.initialize()
        if result:
            print("✅ Chatterbox")
        else:
            print("⚠️  Fallback")
        await tts.cleanup()
    except Exception as e:
        print(f"❌ {e}")

    # LLM
    print("  LLM (OpenAI)...", end=" ", flush=True)
    try:
        from src.pipeline.llm_module import LLMModule
        llm = LLMModule(api_key="test", model="gpt-4o-mini", provider="openai")
        await llm.initialize()
        print("✅")
        await llm.cleanup()
    except Exception as e:
        print(f"❌ {e}")

    # LiveKit
    print("  LiveKit...", end=" ", flush=True)
    try:
        from src.services.livekit_service import LiveKitService
        lk = LiveKitService(api_key="test", api_secret="test", url="wss://test")
        await lk.initialize()
        token = lk.create_token("test", "user")
        if token:
            print("✅")
        else:
            print("❌")
        await lk.cleanup()
    except Exception as e:
        print(f"❌ {e}")


def main():
    print("\n" + "=" * 50)
    print("    AI Avatar 환경 체크")
    print("=" * 50)

    # 모듈 체크
    modules = check_python_modules()

    # 모델 체크
    models = check_model_files()

    # 환경 파일 체크
    env_ok = check_env_file()

    # 디렉토리 체크
    check_directories()

    # 핵심 모듈 테스트
    try:
        asyncio.run(check_core_modules())
    except Exception as e:
        print(f"\n핵심 모듈 테스트 실패: {e}")

    # 요약
    print_header("요약")

    total_modules = len(modules)
    ok_modules = sum(modules.values())

    total_models = len(models)
    ok_models = sum(models.values())

    print(f"  Python 모듈: {ok_modules}/{total_modules}")
    print(f"  모델 파일: {ok_models}/{total_models}")
    print(f"  환경 설정: {'✅' if env_ok else '⚠️  API 키 설정 필요'}")

    if ok_modules == total_modules and ok_models == total_models:
        print("\n🎉 모든 체크 통과! 서버 실행 가능")
        return 0
    else:
        print("\n⚠️  일부 항목 확인 필요")
        return 1


if __name__ == "__main__":
    sys.exit(main())
