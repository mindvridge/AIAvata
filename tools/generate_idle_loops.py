#!/usr/bin/env python3
"""
Idle Loop Generation Tool.

LivePortrait + MediaPipe를 사용하여 idle 루프 영상 생성

사용법:
    python tools/generate_idle_loops.py --image avatar.jpg --output assets/idle_loops/

주의: InsightFace 대신 MediaPipe 사용 (상업적 라이선스)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# 감정별 설정
EMOTION_CONFIGS = {
    "neutral": {
        "filename": "neutral_idle.mp4",
        "blink_frequency": 3.0,  # 초당 깜빡임
        "head_movement": 0.005,  # 머리 움직임 강도
        "expression_intensity": 0.0,
    },
    "happy": {
        "filename": "happy_smile.mp4",
        "blink_frequency": 4.0,
        "head_movement": 0.01,
        "expression_intensity": 0.3,
        "mouth_curve": 0.2,  # 미소
    },
    "sad": {
        "filename": "sad_idle.mp4",
        "blink_frequency": 2.0,
        "head_movement": 0.003,
        "expression_intensity": -0.2,
        "head_tilt": -5,  # 고개 숙임
    },
    "listening": {
        "filename": "listening_nod.mp4",
        "blink_frequency": 3.0,
        "head_movement": 0.015,  # 고개 끄덕임
        "nod_frequency": 1.5,
        "expression_intensity": 0.1,
    },
    "thinking": {
        "filename": "thinking.mp4",
        "blink_frequency": 2.5,
        "head_movement": 0.008,
        "look_direction": "up_left",  # 시선
        "expression_intensity": 0.05,
    },
    "concerned": {
        "filename": "concerned.mp4",
        "blink_frequency": 3.5,
        "head_movement": 0.007,
        "eyebrow_raise": 0.15,  # 눈썹 올림
        "expression_intensity": -0.1,
    },
    "sympathetic": {
        "filename": "sympathetic.mp4",
        "blink_frequency": 2.5,
        "head_movement": 0.006,
        "head_tilt": 5,  # 고개 기울임
        "expression_intensity": 0.15,
    },
}


class IdleLoopGenerator:
    """Idle 루프 영상 생성기"""

    def __init__(
        self,
        output_size: Tuple[int, int] = (512, 512),
        fps: int = 30,
        duration: float = 5.0,
    ):
        """
        Initialize Idle Loop Generator.

        Args:
            output_size: 출력 영상 크기 (width, height)
            fps: 프레임 레이트
            duration: 루프 길이 (초)
        """
        self.output_size = output_size
        self.fps = fps
        self.duration = duration
        self.total_frames = int(fps * duration)

        self._face_mesh = None
        self._live_portrait = None

    def initialize(self) -> bool:
        """모델 초기화"""
        # MediaPipe Face Mesh 초기화
        try:
            import mediapipe as mp

            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            logger.info("MediaPipe Face Mesh initialized")
        except ImportError:
            logger.error("MediaPipe not installed. Install with: pip install mediapipe")
            return False

        # LivePortrait 초기화 (선택적)
        try:
            # LivePortrait는 별도 설치 필요
            # from liveportrait import LivePortrait
            # self._live_portrait = LivePortrait()
            logger.info("LivePortrait: Using placeholder (install separately)")
        except ImportError:
            logger.warning("LivePortrait not available. Using simple animation.")

        return True

    def load_source_image(self, image_path: str) -> Optional[np.ndarray]:
        """소스 이미지 로드"""
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None

        # 크기 조정
        image = cv2.resize(image, self.output_size)
        logger.info(f"Loaded source image: {image_path}")

        return image

    def detect_face(self, image: np.ndarray) -> Optional[dict]:
        """얼굴 랜드마크 감지"""
        if self._face_mesh is None:
            return None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            logger.warning("No face detected in image")
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]

        return {
            "landmarks": [
                {"x": lm.x * w, "y": lm.y * h, "z": lm.z}
                for lm in landmarks.landmark
            ]
        }

    def generate_emotion_loop(
        self,
        source_image: np.ndarray,
        emotion: str,
        output_path: str,
    ) -> bool:
        """
        특정 감정의 idle 루프 생성

        Args:
            source_image: 소스 이미지
            emotion: 감정 이름
            output_path: 출력 파일 경로

        Returns:
            성공 여부
        """
        config = EMOTION_CONFIGS.get(emotion, EMOTION_CONFIGS["neutral"])

        logger.info(f"Generating {emotion} idle loop...")

        # 비디오 라이터 초기화
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            self.output_size,
        )

        # 프레임 생성
        for frame_idx in range(self.total_frames):
            t = frame_idx / self.fps  # 현재 시간 (초)

            # 애니메이션 적용
            animated_frame = self._apply_animation(
                source_image.copy(),
                t,
                config,
            )

            out.write(animated_frame)

        out.release()
        logger.info(f"Generated: {output_path}")

        return True

    def _apply_animation(
        self,
        image: np.ndarray,
        t: float,
        config: dict,
    ) -> np.ndarray:
        """
        애니메이션 효과 적용

        실제 구현에서는 LivePortrait를 사용하여 더 자연스러운 애니메이션 생성
        여기서는 간단한 이미지 변형으로 대체
        """
        # 눈 깜빡임 시뮬레이션
        blink_freq = config.get("blink_frequency", 3.0)
        blink_phase = (t * blink_freq) % 1.0

        # 머리 움직임
        head_movement = config.get("head_movement", 0.005)
        dx = int(np.sin(t * 2 * np.pi / self.duration) * image.shape[1] * head_movement)
        dy = int(np.cos(t * 1.5 * np.pi / self.duration) * image.shape[0] * head_movement)

        # 이미지 변환 매트릭스
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        animated = cv2.warpAffine(
            image,
            M,
            self.output_size,
            borderMode=cv2.BORDER_REFLECT,
        )

        # 고개 끄덕임 (listening)
        if "nod_frequency" in config:
            nod_freq = config["nod_frequency"]
            nod_angle = np.sin(t * nod_freq * 2 * np.pi) * 3  # ±3도
            center = (self.output_size[0] // 2, self.output_size[1] // 2)
            R = cv2.getRotationMatrix2D(center, nod_angle, 1.0)
            animated = cv2.warpAffine(
                animated,
                R,
                self.output_size,
                borderMode=cv2.BORDER_REFLECT,
            )

        # 밝기 미세 조정 (호흡 효과)
        brightness_var = 1.0 + 0.02 * np.sin(t * 0.5 * np.pi)
        animated = np.clip(animated * brightness_var, 0, 255).astype(np.uint8)

        return animated

    def generate_all_loops(
        self,
        source_image_path: str,
        output_dir: str,
        emotions: Optional[List[str]] = None,
    ) -> dict:
        """
        모든 감정의 idle 루프 생성

        Args:
            source_image_path: 소스 이미지 경로
            output_dir: 출력 디렉토리
            emotions: 생성할 감정 목록 (없으면 전체)

        Returns:
            생성 결과 {emotion: success}
        """
        # 초기화
        if not self.initialize():
            return {}

        # 소스 이미지 로드
        source_image = self.load_source_image(source_image_path)
        if source_image is None:
            return {}

        # 얼굴 감지
        face_data = self.detect_face(source_image)
        if face_data is None:
            logger.warning("Proceeding without face landmarks")

        # 출력 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 감정 목록
        if emotions is None:
            emotions = list(EMOTION_CONFIGS.keys())

        # 각 감정별 루프 생성
        results = {}
        for emotion in emotions:
            if emotion not in EMOTION_CONFIGS:
                logger.warning(f"Unknown emotion: {emotion}")
                results[emotion] = False
                continue

            config = EMOTION_CONFIGS[emotion]
            out_file = output_path / config["filename"]

            success = self.generate_emotion_loop(
                source_image,
                emotion,
                str(out_file),
            )
            results[emotion] = success

        return results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Generate idle loop videos for avatar animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 모든 감정의 idle 루프 생성
    python tools/generate_idle_loops.py --image avatar.jpg --output assets/idle_loops/

    # 특정 감정만 생성
    python tools/generate_idle_loops.py --image avatar.jpg --output assets/idle_loops/ --emotions neutral happy

    # 고해상도로 생성
    python tools/generate_idle_loops.py --image avatar.jpg --output assets/idle_loops/ --size 1024
        """,
    )

    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Source avatar image path",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="assets/idle_loops/",
        help="Output directory (default: assets/idle_loops/)",
    )
    parser.add_argument(
        "--emotions",
        "-e",
        nargs="+",
        choices=list(EMOTION_CONFIGS.keys()),
        help="Specific emotions to generate (default: all)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=5.0,
        help="Loop duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate (default: 30)",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=512,
        help="Output video size (default: 512)",
    )

    args = parser.parse_args()

    # 소스 이미지 확인
    if not Path(args.image).exists():
        logger.error(f"Source image not found: {args.image}")
        sys.exit(1)

    # 생성기 초기화
    generator = IdleLoopGenerator(
        output_size=(args.size, args.size),
        fps=args.fps,
        duration=args.duration,
    )

    # 루프 생성
    results = generator.generate_all_loops(
        source_image_path=args.image,
        output_dir=args.output,
        emotions=args.emotions,
    )

    # 결과 출력
    print("\n" + "=" * 50)
    print("Generation Results:")
    print("=" * 50)

    success_count = 0
    for emotion, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {emotion}")
        if success:
            success_count += 1

    print("=" * 50)
    print(f"Total: {success_count}/{len(results)} successful")

    if success_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
