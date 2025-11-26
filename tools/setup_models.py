#!/usr/bin/env python3
"""
Model Setup Tool.

필요한 모델들을 다운로드하고 설정합니다.

사용법:
    python tools/setup_models.py
    python tools/setup_models.py --models stt tts
    python tools/setup_models.py --device cuda
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


MODEL_CONFIGS = {
    "stt": {
        "name": "SenseVoice-Small",
        "description": "FunAudioLLM/SenseVoiceSmall - STT + 감정인식",
        "package": "funasr",
        "model_id": "FunAudioLLM/SenseVoiceSmall",
        "license": "Apache 2.0",
    },
    "tts": {
        "name": "Chatterbox TTS",
        "description": "Resemble AI Chatterbox - 고품질 TTS",
        "package": "chatterbox-tts",
        "license": "MIT",
    },
    "vad": {
        "name": "Silero VAD",
        "description": "Silero Voice Activity Detection",
        "package": "silero-vad",
        "repo": "snakers4/silero-vad",
        "license": "MIT",
    },
    "face": {
        "name": "MediaPipe Face Mesh",
        "description": "얼굴 랜드마크 감지 (InsightFace 대체)",
        "package": "mediapipe",
        "license": "Apache 2.0",
    },
    "lipsync": {
        "name": "MuseTalk",
        "description": "실시간 립싱크",
        "repo": "https://github.com/TMElyralab/MuseTalk",
        "license": "MIT",
        "model_dir": "models/musetalk",
        "huggingface_repo": "TMElyralab/MuseTalk",
    },
    "portrait": {
        "name": "LivePortrait",
        "description": "Idle 루프 및 얼굴 애니메이션 생성",
        "repo": "https://github.com/KwaiVGI/LivePortrait",
        "license": "MIT",
        "model_dir": "models/live_portrait",
        "huggingface_repo": "KwaiVGI/LivePortrait",
    },
}


def check_gpu() -> dict:
    """GPU 상태 확인"""
    result = {
        "available": False,
        "device_count": 0,
        "devices": [],
    }

    try:
        import torch

        result["available"] = torch.cuda.is_available()
        if result["available"]:
            result["device_count"] = torch.cuda.device_count()
            for i in range(result["device_count"]):
                result["devices"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9,
                })
    except ImportError:
        pass

    return result


def setup_stt(device: str = "cuda") -> bool:
    """SenseVoice STT 모델 설정"""
    logger.info("Setting up SenseVoice-Small STT model...")

    try:
        from funasr import AutoModel

        # 모델 다운로드 및 초기화
        model = AutoModel(
            model="FunAudioLLM/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True,
        )

        logger.info("SenseVoice-Small model downloaded and initialized")
        del model
        return True

    except ImportError:
        logger.error("funasr not installed. Install with: pip install funasr")
        return False

    except Exception as e:
        logger.error(f"Failed to setup STT model: {e}")
        return False


def setup_tts(device: str = "cuda") -> bool:
    """Chatterbox TTS 모델 설정"""
    logger.info("Setting up Chatterbox TTS model...")

    try:
        from chatterbox.tts import ChatterboxTTS

        # 모델 다운로드
        model = ChatterboxTTS.from_pretrained(device=device)

        logger.info("Chatterbox TTS model downloaded and initialized")
        del model
        return True

    except ImportError:
        logger.error("chatterbox-tts not installed. Install with: pip install chatterbox-tts")
        return False

    except Exception as e:
        logger.error(f"Failed to setup TTS model: {e}")
        return False


def setup_vad() -> bool:
    """Silero VAD 모델 설정"""
    logger.info("Setting up Silero VAD model...")

    try:
        import torch

        # Silero VAD 다운로드
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        logger.info("Silero VAD model downloaded")
        return True

    except ImportError:
        logger.error("torch not installed. Install with: pip install torch")
        return False

    except Exception as e:
        logger.error(f"Failed to setup VAD model: {e}")
        return False


def setup_face() -> bool:
    """MediaPipe Face Mesh 설정"""
    logger.info("Setting up MediaPipe Face Mesh...")

    try:
        import mediapipe as mp

        # Face Mesh 초기화 (모델 다운로드)
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        face_mesh.close()

        logger.info("MediaPipe Face Mesh initialized")
        return True

    except ImportError:
        logger.error("mediapipe not installed. Install with: pip install mediapipe")
        return False

    except Exception as e:
        logger.error(f"Failed to setup MediaPipe: {e}")
        return False


def setup_lipsync(device: str = "cuda") -> bool:
    """MuseTalk 립싱크 모델 설정"""
    logger.info("Setting up MuseTalk lip sync model...")

    model_dir = Path(MODEL_CONFIGS["lipsync"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Hugging Face Hub를 통한 모델 다운로드 시도
        try:
            from huggingface_hub import snapshot_download

            logger.info("Downloading MuseTalk models from Hugging Face...")
            snapshot_download(
                repo_id="TMElyralab/MuseTalk",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt", ".git*"],
            )
            logger.info(f"MuseTalk models downloaded to {model_dir}")
            return True

        except ImportError:
            logger.warning("huggingface_hub not installed. Trying alternative method...")

        # 대안: MuseTalk 패키지 사용
        try:
            from musetalk.models.unet import MuseTalkUNet
            logger.info("MuseTalk package is installed")
            return True
        except ImportError:
            pass

        # 모델 디렉토리만 생성
        logger.warning(
            f"MuseTalk model not downloaded. "
            f"Install huggingface_hub: pip install huggingface_hub\n"
            f"Or manually clone: git clone https://github.com/TMElyralab/MuseTalk"
        )
        return False

    except Exception as e:
        logger.error(f"Failed to setup MuseTalk: {e}")
        return False


def setup_portrait(device: str = "cuda") -> bool:
    """LivePortrait 얼굴 애니메이션 모델 설정"""
    logger.info("Setting up LivePortrait model...")

    model_dir = Path(MODEL_CONFIGS["portrait"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Hugging Face Hub를 통한 모델 다운로드 시도
        try:
            from huggingface_hub import snapshot_download

            logger.info("Downloading LivePortrait models from Hugging Face...")
            snapshot_download(
                repo_id="KwaiVGI/LivePortrait",
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                ignore_patterns=["*.md", "*.txt", ".git*", "docs/*", "assets/*"],
            )
            logger.info(f"LivePortrait models downloaded to {model_dir}")
            return True

        except ImportError:
            logger.warning("huggingface_hub not installed. Trying alternative method...")

        # 대안: LivePortrait 패키지 사용
        try:
            from liveportrait.inference import LivePortraitInference
            logger.info("LivePortrait package is installed")
            return True
        except ImportError:
            pass

        # 모델 디렉토리만 생성
        logger.warning(
            f"LivePortrait model not downloaded. "
            f"Install huggingface_hub: pip install huggingface_hub\n"
            f"Or manually clone: git clone https://github.com/KwaiVGI/LivePortrait"
        )
        return False

    except Exception as e:
        logger.error(f"Failed to setup LivePortrait: {e}")
        return False


def setup_directories() -> None:
    """필요한 디렉토리 생성"""
    directories = [
        "assets/avatars",
        "assets/idle_loops",
        "assets/voice_samples",
        "models",
        "logs",
    ]

    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")


def print_model_info() -> None:
    """모델 정보 출력"""
    print("\n" + "=" * 60)
    print("Model Information")
    print("=" * 60)

    for model_id, config in MODEL_CONFIGS.items():
        print(f"\n{config['name']}:")
        print(f"  Description: {config['description']}")
        print(f"  License: {config['license']}")
        if "repo" in config:
            print(f"  Repository: {config['repo']}")
        if "huggingface_repo" in config:
            print(f"  HuggingFace: https://huggingface.co/{config['huggingface_repo']}")

    print("\n" + "=" * 60)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Download and setup required models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        choices=["stt", "tts", "vad", "face", "lipsync", "portrait", "all"],
        default=["all"],
        help="Models to setup (default: all)",
    )
    parser.add_argument(
        "--device",
        "-d",
        choices=["cuda", "cpu", "mps"],
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip GPU availability check",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Realtime AI Avatar - Model Setup")
    print("=" * 60)

    # GPU 확인
    if not args.skip_gpu_check:
        gpu_info = check_gpu()
        if gpu_info["available"]:
            print(f"\nGPU Available: Yes ({gpu_info['device_count']} device(s))")
            for dev in gpu_info["devices"]:
                print(f"  - {dev['name']} ({dev['memory_gb']:.1f} GB)")
        else:
            print("\nGPU Available: No (using CPU)")
            if args.device == "cuda":
                args.device = "cpu"
                logger.warning("Falling back to CPU")
    else:
        print("\nGPU check skipped")

    # 디렉토리 생성
    print("\nCreating directories...")
    setup_directories()

    # 모델 설정
    models_to_setup = args.models
    if "all" in models_to_setup:
        models_to_setup = ["stt", "tts", "vad", "face", "lipsync", "portrait"]

    results = {}

    print(f"\nSetting up models: {', '.join(models_to_setup)}")
    print("-" * 60)

    for model in models_to_setup:
        if model == "stt":
            results["stt"] = setup_stt(args.device)
        elif model == "tts":
            results["tts"] = setup_tts(args.device)
        elif model == "vad":
            results["vad"] = setup_vad()
        elif model == "face":
            results["face"] = setup_face()
        elif model == "lipsync":
            results["lipsync"] = setup_lipsync(args.device)
        elif model == "portrait":
            results["portrait"] = setup_portrait(args.device)

    # 결과 출력
    print("\n" + "=" * 60)
    print("Setup Results:")
    print("=" * 60)

    success_count = 0
    for model, success in results.items():
        status = "✓" if success else "✗"
        config = MODEL_CONFIGS.get(model, {})
        name = config.get("name", model)
        print(f"  {status} {name}")
        if success:
            success_count += 1

    print("-" * 60)
    print(f"Total: {success_count}/{len(results)} successful")

    # 모델 정보 출력
    print_model_info()

    # 다음 단계 안내
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Add avatar image to: assets/avatars/")
    print("2. Generate idle loops: python tools/generate_idle_loops.py --image <path>")
    print("3. Configure environment: cp .env.example .env")
    print("4. Start server: uvicorn src.main:app --reload")
    print("=" * 60)

    if success_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
