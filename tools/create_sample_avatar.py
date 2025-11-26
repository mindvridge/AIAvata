#!/usr/bin/env python3
"""
Sample Avatar Image Creator.

테스트용 샘플 아바타 이미지 생성

사용법:
    python tools/create_sample_avatar.py
    python tools/create_sample_avatar.py --output assets/avatars/custom.png
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_gradient_background(width: int, height: int) -> np.ndarray:
    """그라데이션 배경 생성"""
    # 상단: 밝은 파란색, 하단: 진한 파란색
    top_color = np.array([200, 180, 150])  # BGR
    bottom_color = np.array([80, 60, 40])

    background = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        ratio = y / height
        color = top_color * (1 - ratio) + bottom_color * ratio
        background[y, :] = color.astype(np.uint8)

    return background


def draw_face_shape(image: np.ndarray, center: tuple, size: int) -> np.ndarray:
    """얼굴 형태 그리기"""
    cx, cy = center

    # 얼굴 타원
    face_color = (210, 195, 180)  # 살색 (BGR)
    cv2.ellipse(
        image,
        (cx, cy + 10),
        (size // 2 - 20, size // 2 + 10),
        0, 0, 360,
        face_color,
        -1,
        cv2.LINE_AA,
    )

    return image


def draw_hair(image: np.ndarray, center: tuple, size: int) -> np.ndarray:
    """머리카락 그리기"""
    cx, cy = center
    hair_color = (40, 30, 25)  # 어두운 갈색 (BGR)

    # 상단 머리카락
    cv2.ellipse(
        image,
        (cx, cy - 30),
        (size // 2 + 10, size // 3),
        0, 180, 360,
        hair_color,
        -1,
        cv2.LINE_AA,
    )

    # 양쪽 머리카락
    cv2.ellipse(
        image,
        (cx - size // 2 + 30, cy + 20),
        (35, 80),
        0, 0, 360,
        hair_color,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (cx + size // 2 - 30, cy + 20),
        (35, 80),
        0, 0, 360,
        hair_color,
        -1,
        cv2.LINE_AA,
    )

    return image


def draw_eyes(image: np.ndarray, center: tuple, size: int) -> np.ndarray:
    """눈 그리기"""
    cx, cy = center
    eye_y = cy - 10
    eye_spacing = size // 5

    # 눈 흰자
    white_color = (255, 255, 255)
    cv2.ellipse(
        image,
        (cx - eye_spacing, eye_y),
        (25, 18),
        0, 0, 360,
        white_color,
        -1,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (cx + eye_spacing, eye_y),
        (25, 18),
        0, 0, 360,
        white_color,
        -1,
        cv2.LINE_AA,
    )

    # 홍채
    iris_color = (80, 60, 40)  # 갈색
    cv2.circle(image, (cx - eye_spacing, eye_y), 12, iris_color, -1, cv2.LINE_AA)
    cv2.circle(image, (cx + eye_spacing, eye_y), 12, iris_color, -1, cv2.LINE_AA)

    # 동공
    pupil_color = (20, 15, 10)
    cv2.circle(image, (cx - eye_spacing, eye_y), 6, pupil_color, -1, cv2.LINE_AA)
    cv2.circle(image, (cx + eye_spacing, eye_y), 6, pupil_color, -1, cv2.LINE_AA)

    # 눈 반짝임
    highlight_color = (255, 255, 255)
    cv2.circle(image, (cx - eye_spacing - 4, eye_y - 4), 3, highlight_color, -1, cv2.LINE_AA)
    cv2.circle(image, (cx + eye_spacing - 4, eye_y - 4), 3, highlight_color, -1, cv2.LINE_AA)

    # 눈썹
    eyebrow_color = (50, 40, 35)
    cv2.ellipse(
        image,
        (cx - eye_spacing, eye_y - 30),
        (30, 8),
        0, 180, 360,
        eyebrow_color,
        3,
        cv2.LINE_AA,
    )
    cv2.ellipse(
        image,
        (cx + eye_spacing, eye_y - 30),
        (30, 8),
        0, 180, 360,
        eyebrow_color,
        3,
        cv2.LINE_AA,
    )

    return image


def draw_nose(image: np.ndarray, center: tuple) -> np.ndarray:
    """코 그리기"""
    cx, cy = center
    nose_color = (190, 175, 160)

    # 코 라인
    pts = np.array([
        [cx, cy + 10],
        [cx - 8, cy + 40],
        [cx, cy + 45],
        [cx + 8, cy + 40],
    ], np.int32)
    cv2.polylines(image, [pts], False, nose_color, 2, cv2.LINE_AA)

    return image


def draw_mouth(image: np.ndarray, center: tuple, expression: str = "smile") -> np.ndarray:
    """입 그리기"""
    cx, cy = center
    mouth_y = cy + 70

    if expression == "smile":
        # 미소
        lip_color = (130, 130, 180)  # 입술색
        cv2.ellipse(
            image,
            (cx, mouth_y),
            (35, 15),
            0, 0, 180,
            lip_color,
            -1,
            cv2.LINE_AA,
        )
        # 윗입술 라인
        cv2.ellipse(
            image,
            (cx, mouth_y - 5),
            (30, 5),
            0, 0, 180,
            lip_color,
            -1,
            cv2.LINE_AA,
        )
    else:
        # 중립
        lip_color = (130, 130, 180)
        cv2.ellipse(
            image,
            (cx, mouth_y),
            (25, 8),
            0, 0, 360,
            lip_color,
            -1,
            cv2.LINE_AA,
        )

    return image


def draw_neck_and_shoulders(image: np.ndarray, center: tuple, size: int) -> np.ndarray:
    """목과 어깨 그리기"""
    cx, cy = center
    neck_color = (200, 185, 170)
    shirt_color = (80, 80, 120)  # 어두운 파란색 옷

    # 목
    cv2.rectangle(
        image,
        (cx - 30, cy + size // 2 - 30),
        (cx + 30, cy + size // 2 + 40),
        neck_color,
        -1,
    )

    # 어깨/옷
    pts = np.array([
        [cx - 150, cy + size // 2 + 100],
        [cx - 60, cy + size // 2 + 20],
        [cx + 60, cy + size // 2 + 20],
        [cx + 150, cy + size // 2 + 100],
    ], np.int32)
    cv2.fillPoly(image, [pts], shirt_color)

    return image


def create_avatar_image(
    width: int = 512,
    height: int = 512,
    expression: str = "smile",
) -> np.ndarray:
    """
    샘플 아바타 이미지 생성

    Args:
        width: 이미지 너비
        height: 이미지 높이
        expression: 표정 ('smile' 또는 'neutral')

    Returns:
        아바타 이미지 (BGR)
    """
    # 배경 생성
    image = create_gradient_background(width, height)

    center = (width // 2, height // 2 - 30)
    face_size = min(width, height) - 100

    # 각 요소 그리기 (순서 중요)
    image = draw_neck_and_shoulders(image, center, face_size)
    image = draw_hair(image, center, face_size)
    image = draw_face_shape(image, center, face_size)
    image = draw_eyes(image, center, face_size)
    image = draw_nose(image, center)
    image = draw_mouth(image, center, expression)

    # 약간의 블러로 자연스럽게
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Create sample avatar image for testing",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="assets/avatars/sample_avatar.png",
        help="Output image path (default: assets/avatars/sample_avatar.png)",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=512,
        help="Image size (default: 512)",
    )
    parser.add_argument(
        "--expression",
        "-e",
        choices=["smile", "neutral"],
        default="smile",
        help="Facial expression (default: smile)",
    )

    args = parser.parse_args()

    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 아바타 생성
    logger.info(f"Creating sample avatar ({args.size}x{args.size})...")
    avatar = create_avatar_image(
        width=args.size,
        height=args.size,
        expression=args.expression,
    )

    # 저장
    cv2.imwrite(str(output_path), avatar)
    logger.info(f"Saved avatar to: {output_path}")

    print(f"\nSample avatar created: {output_path}")
    print(f"Size: {args.size}x{args.size}")
    print(f"Expression: {args.expression}")
    print("\nNext steps:")
    print(f"  1. Generate idle loops: python tools/generate_idle_loops.py --image {output_path}")
    print("  2. Or use your own avatar image")


if __name__ == "__main__":
    main()
