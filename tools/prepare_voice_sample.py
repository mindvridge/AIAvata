#!/usr/bin/env python3
"""
Voice Sample Preparation Tool.

TTS 음성 클로닝용 음성 샘플 준비

사용법:
    python tools/prepare_voice_sample.py --input voice.wav --output assets/voice_samples/custom.wav
    python tools/prepare_voice_sample.py --record --duration 10 --output assets/voice_samples/recorded.wav
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 권장 음성 샘플 설정
RECOMMENDED_SAMPLE_RATE = 24000
RECOMMENDED_DURATION_MIN = 5.0  # 최소 5초
RECOMMENDED_DURATION_MAX = 30.0  # 최대 30초


def load_audio(file_path: str, target_sr: int = RECOMMENDED_SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    오디오 파일 로드 및 전처리

    Args:
        file_path: 오디오 파일 경로
        target_sr: 목표 샘플레이트

    Returns:
        오디오 데이터 (float32, mono)
    """
    try:
        import soundfile as sf

        audio, sr = sf.read(file_path)

        # 스테레오 → 모노
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # 리샘플링
        if sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                logger.info(f"Resampled from {sr}Hz to {target_sr}Hz")
            except ImportError:
                logger.warning("librosa not installed. Skipping resampling.")

        return audio.astype(np.float32)

    except ImportError:
        logger.error("soundfile not installed. Install with: pip install soundfile")
        return None
    except Exception as e:
        logger.error(f"Failed to load audio: {e}")
        return None


def save_audio(audio: np.ndarray, file_path: str, sample_rate: int = RECOMMENDED_SAMPLE_RATE) -> bool:
    """
    오디오 저장

    Args:
        audio: 오디오 데이터
        file_path: 출력 경로
        sample_rate: 샘플레이트

    Returns:
        성공 여부
    """
    try:
        import soundfile as sf

        sf.write(file_path, audio, sample_rate, subtype="PCM_16")
        logger.info(f"Saved audio to: {file_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        return False


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    오디오 정규화 (볼륨 조정)

    Args:
        audio: 오디오 데이터
        target_db: 목표 데시벨

    Returns:
        정규화된 오디오
    """
    # RMS 계산
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio

    # 현재 dB
    current_db = 20 * np.log10(rms)

    # 게인 계산 및 적용
    gain = 10 ** ((target_db - current_db) / 20)
    normalized = audio * gain

    # 클리핑 방지
    max_val = np.max(np.abs(normalized))
    if max_val > 0.99:
        normalized = normalized * 0.99 / max_val

    return normalized.astype(np.float32)


def remove_silence(
    audio: np.ndarray,
    sample_rate: int = RECOMMENDED_SAMPLE_RATE,
    threshold_db: float = -40.0,
    min_silence_duration: float = 0.3,
) -> np.ndarray:
    """
    시작/끝 무음 제거

    Args:
        audio: 오디오 데이터
        sample_rate: 샘플레이트
        threshold_db: 무음 임계값 (dB)
        min_silence_duration: 최소 무음 길이 (초)

    Returns:
        무음 제거된 오디오
    """
    # 프레임 단위 에너지 계산
    frame_size = int(sample_rate * 0.02)  # 20ms
    hop_size = frame_size // 2

    threshold = 10 ** (threshold_db / 20)

    # 시작점 찾기
    start_idx = 0
    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > threshold:
            start_idx = max(0, i - int(sample_rate * 0.1))  # 100ms 여유
            break

    # 끝점 찾기
    end_idx = len(audio)
    for i in range(len(audio) - frame_size, 0, -hop_size):
        frame = audio[i:i + frame_size]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms > threshold:
            end_idx = min(len(audio), i + frame_size + int(sample_rate * 0.1))
            break

    trimmed = audio[start_idx:end_idx]
    logger.info(f"Trimmed silence: {len(audio) / sample_rate:.2f}s -> {len(trimmed) / sample_rate:.2f}s")

    return trimmed


def validate_audio(
    audio: np.ndarray,
    sample_rate: int = RECOMMENDED_SAMPLE_RATE,
) -> dict:
    """
    오디오 품질 검증

    Args:
        audio: 오디오 데이터
        sample_rate: 샘플레이트

    Returns:
        검증 결과
    """
    duration = len(audio) / sample_rate
    rms = np.sqrt(np.mean(audio ** 2))
    peak = np.max(np.abs(audio))
    db = 20 * np.log10(rms) if rms > 0 else -100

    result = {
        "duration": duration,
        "sample_rate": sample_rate,
        "rms_db": db,
        "peak": peak,
        "is_valid": True,
        "warnings": [],
    }

    # 길이 검사
    if duration < RECOMMENDED_DURATION_MIN:
        result["warnings"].append(f"Duration too short: {duration:.1f}s (min: {RECOMMENDED_DURATION_MIN}s)")
    if duration > RECOMMENDED_DURATION_MAX:
        result["warnings"].append(f"Duration too long: {duration:.1f}s (max: {RECOMMENDED_DURATION_MAX}s)")

    # 볼륨 검사
    if db < -35:
        result["warnings"].append(f"Audio too quiet: {db:.1f}dB (recommended: > -35dB)")
    if peak > 0.99:
        result["warnings"].append("Audio may be clipped")

    result["is_valid"] = len(result["warnings"]) == 0

    return result


def record_audio(
    duration: float,
    sample_rate: int = RECOMMENDED_SAMPLE_RATE,
) -> Optional[np.ndarray]:
    """
    마이크로 음성 녹음

    Args:
        duration: 녹음 시간 (초)
        sample_rate: 샘플레이트

    Returns:
        녹음된 오디오
    """
    try:
        import sounddevice as sd

        logger.info(f"Recording for {duration} seconds...")
        logger.info("Speak now!")

        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()

        logger.info("Recording complete")
        return audio.flatten()

    except ImportError:
        logger.error("sounddevice not installed. Install with: pip install sounddevice")
        return None
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        return None


def generate_sample_script() -> str:
    """TTS 음성 샘플 녹음용 스크립트 생성"""
    return """
========================================
음성 샘플 녹음 가이드
========================================

다음 문장을 자연스럽게 읽어주세요:

1. "안녕하세요, 반갑습니다."
2. "오늘 날씨가 정말 좋네요."
3. "어떻게 도와드릴까요?"
4. "잠시만 기다려 주세요."
5. "감사합니다. 좋은 하루 되세요."

Tips:
- 조용한 환경에서 녹음하세요
- 마이크와 적당한 거리(15-30cm)를 유지하세요
- 자연스러운 속도로 말해주세요
- 최소 5초, 최대 30초 분량이 적당합니다

========================================
"""


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Prepare voice sample for TTS cloning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 기존 오디오 파일 처리
    python tools/prepare_voice_sample.py --input voice.wav --output assets/voice_samples/custom.wav

    # 마이크로 녹음
    python tools/prepare_voice_sample.py --record --duration 10 --output assets/voice_samples/recorded.wav

    # 검증만 수행
    python tools/prepare_voice_sample.py --input voice.wav --validate-only
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        help="Input audio file path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="assets/voice_samples/voice.wav",
        help="Output audio file path",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record from microphone",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate input, don't process",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip volume normalization",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Skip silence trimming",
    )
    parser.add_argument(
        "--show-script",
        action="store_true",
        help="Show recording script guide",
    )

    args = parser.parse_args()

    # 스크립트 가이드 표시
    if args.show_script:
        print(generate_sample_script())
        return

    # 입력 확인
    if not args.input and not args.record:
        print("Error: Either --input or --record is required")
        print("Use --show-script to see recording guide")
        parser.print_help()
        sys.exit(1)

    # 오디오 로드 또는 녹음
    if args.record:
        print(generate_sample_script())
        input("Press Enter to start recording...")
        audio = record_audio(args.duration)
    else:
        if not Path(args.input).exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        audio = load_audio(args.input)

    if audio is None:
        sys.exit(1)

    # 검증
    validation = validate_audio(audio)

    print("\n" + "=" * 50)
    print("Audio Analysis:")
    print("=" * 50)
    print(f"  Duration: {validation['duration']:.2f} seconds")
    print(f"  Sample Rate: {validation['sample_rate']} Hz")
    print(f"  RMS Level: {validation['rms_db']:.1f} dB")
    print(f"  Peak: {validation['peak']:.3f}")

    if validation["warnings"]:
        print("\nWarnings:")
        for warning in validation["warnings"]:
            print(f"  ! {warning}")

    if args.validate_only:
        sys.exit(0 if validation["is_valid"] else 1)

    # 처리
    processed = audio

    if not args.no_trim:
        processed = remove_silence(processed)

    if not args.no_normalize:
        processed = normalize_audio(processed)

    # 재검증
    final_validation = validate_audio(processed)

    # 출력 저장
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if save_audio(processed, str(output_path)):
        print("\n" + "=" * 50)
        print("Processing Complete:")
        print("=" * 50)
        print(f"  Output: {output_path}")
        print(f"  Final Duration: {final_validation['duration']:.2f} seconds")
        print(f"  Final Level: {final_validation['rms_db']:.1f} dB")
        print("\nVoice sample is ready for TTS cloning!")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
