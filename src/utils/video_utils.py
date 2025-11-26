"""
Video processing utilities.

비디오 처리 유틸리티 모듈
"""

import asyncio
import logging
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    비디오 처리 유틸리티

    Features:
    - 프레임 추출
    - 크기 조정
    - 인코딩/디코딩
    - 프레임 조작
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        target_fps: int = 30,
        jpeg_quality: int = 85,
    ):
        """
        Initialize Video Processor.

        Args:
            target_size: 목표 프레임 크기 (width, height)
            target_fps: 목표 FPS
            jpeg_quality: JPEG 압축 품질
        """
        self.target_size = target_size
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality

    def load_video(self, video_path: str) -> List[np.ndarray]:
        """
        비디오 파일에서 모든 프레임 로드

        Args:
            video_path: 비디오 파일 경로

        Returns:
            프레임 리스트
        """
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        logger.info(f"Loaded {len(frames)} frames from {video_path}")

        return frames

    def load_video_generator(
        self, video_path: str
    ) -> Generator[np.ndarray, None, None]:
        """
        비디오 프레임 제너레이터 (메모리 효율적)

        Args:
            video_path: 비디오 파일 경로

        Yields:
            비디오 프레임
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

        cap.release()

    def save_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: Optional[int] = None,
        codec: str = "mp4v",
    ) -> bool:
        """
        프레임을 비디오 파일로 저장

        Args:
            frames: 프레임 리스트
            output_path: 출력 파일 경로
            fps: FPS (없으면 기본값 사용)
            codec: 비디오 코덱

        Returns:
            성공 여부
        """
        if not frames:
            logger.error("No frames to save")
            return False

        fps = fps or self.target_fps
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)

        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

        for frame in frames:
            out.write(frame)

        out.release()
        logger.info(f"Saved video: {output_path}")

        return True

    def resize_frame(
        self,
        frame: np.ndarray,
        size: Optional[Tuple[int, int]] = None,
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """
        프레임 크기 조정

        Args:
            frame: 입력 프레임
            size: 목표 크기 (width, height)
            interpolation: 보간 방법

        Returns:
            크기 조정된 프레임
        """
        size = size or self.target_size
        return cv2.resize(frame, size, interpolation=interpolation)

    def encode_frame(
        self,
        frame: np.ndarray,
        format: str = "jpeg",
        quality: Optional[int] = None,
    ) -> bytes:
        """
        프레임을 바이트로 인코딩

        Args:
            frame: 입력 프레임
            format: 인코딩 포맷 (jpeg, png)
            quality: 압축 품질

        Returns:
            인코딩된 바이트
        """
        quality = quality or self.jpeg_quality

        if format.lower() == "jpeg":
            _, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
        elif format.lower() == "png":
            _, encoded = cv2.imencode(
                ".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 9 - quality // 11]
            )
        else:
            _, encoded = cv2.imencode(".jpg", frame)

        return encoded.tobytes()

    def decode_frame(
        self,
        data: bytes,
    ) -> Optional[np.ndarray]:
        """
        바이트를 프레임으로 디코딩

        Args:
            data: 인코딩된 바이트

        Returns:
            디코딩된 프레임 또는 None
        """
        try:
            nparr = np.frombuffer(data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"Failed to decode frame: {e}")
            return None

    def blend_frames(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        두 프레임 블렌딩

        Args:
            frame1: 첫 번째 프레임
            frame2: 두 번째 프레임
            alpha: 블렌딩 비율 (0-1, frame2의 가중치)

        Returns:
            블렌딩된 프레임
        """
        return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)

    def create_crossfade(
        self,
        frames1: List[np.ndarray],
        frames2: List[np.ndarray],
        transition_frames: int = 15,
    ) -> List[np.ndarray]:
        """
        두 프레임 시퀀스 간 크로스페이드

        Args:
            frames1: 첫 번째 시퀀스
            frames2: 두 번째 시퀀스
            transition_frames: 전환 프레임 수

        Returns:
            크로스페이드된 시퀀스
        """
        result = frames1[:-transition_frames].copy()

        for i in range(transition_frames):
            alpha = i / (transition_frames - 1)
            blended = self.blend_frames(
                frames1[-(transition_frames - i)],
                frames2[i],
                alpha,
            )
            result.append(blended)

        result.extend(frames2[transition_frames:])
        return result

    def apply_overlay(
        self,
        background: np.ndarray,
        overlay: np.ndarray,
        position: Tuple[int, int] = (0, 0),
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        오버레이 적용

        Args:
            background: 배경 프레임
            overlay: 오버레이 이미지
            position: 오버레이 위치 (x, y)
            mask: 마스크 (선택적)

        Returns:
            합성된 프레임
        """
        x, y = position
        h, w = overlay.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # 범위 계산
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
        ox1, oy1 = max(0, -x), max(0, -y)
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

        result = background.copy()

        if mask is not None:
            # 마스크를 사용한 블렌딩
            mask_region = mask[oy1:oy2, ox1:ox2]
            if len(mask_region.shape) == 2:
                mask_region = mask_region[:, :, np.newaxis]
            mask_region = mask_region / 255.0

            result[y1:y2, x1:x2] = (
                overlay[oy1:oy2, ox1:ox2] * mask_region
                + background[y1:y2, x1:x2] * (1 - mask_region)
            ).astype(np.uint8)
        else:
            result[y1:y2, x1:x2] = overlay[oy1:oy2, ox1:ox2]

        return result

    def extract_face_region(
        self,
        frame: np.ndarray,
        landmarks: dict,
        padding: float = 0.2,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        얼굴 영역 추출

        Args:
            frame: 입력 프레임
            landmarks: 얼굴 랜드마크
            padding: 패딩 비율

        Returns:
            (얼굴 영역, 바운딩 박스)
        """
        if not landmarks or "landmarks" not in landmarks:
            return frame, (0, 0, frame.shape[1], frame.shape[0])

        points = landmarks["landmarks"]
        xs = [p["x"] for p in points]
        ys = [p["y"] for p in points]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # 패딩 추가
        width = x_max - x_min
        height = y_max - y_min
        x_min = max(0, int(x_min - width * padding))
        y_min = max(0, int(y_min - height * padding))
        x_max = min(frame.shape[1], int(x_max + width * padding))
        y_max = min(frame.shape[0], int(y_max + height * padding))

        face_region = frame[y_min:y_max, x_min:x_max]
        bbox = (x_min, y_min, x_max, y_max)

        return face_region, bbox

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: dict,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 2,
    ) -> np.ndarray:
        """
        랜드마크 시각화

        Args:
            frame: 입력 프레임
            landmarks: 랜드마크 딕셔너리
            color: 점 색상 (BGR)
            radius: 점 반지름

        Returns:
            랜드마크가 그려진 프레임
        """
        result = frame.copy()

        if not landmarks or "landmarks" not in landmarks:
            return result

        for point in landmarks["landmarks"]:
            x, y = int(point["x"]), int(point["y"])
            cv2.circle(result, (x, y), radius, color, -1)

        return result

    def convert_color(
        self,
        frame: np.ndarray,
        from_space: str = "BGR",
        to_space: str = "RGB",
    ) -> np.ndarray:
        """
        색상 공간 변환

        Args:
            frame: 입력 프레임
            from_space: 입력 색상 공간
            to_space: 출력 색상 공간

        Returns:
            변환된 프레임
        """
        conversion_map = {
            ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
            ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
            ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
            ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
            ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
            ("GRAY", "RGB"): cv2.COLOR_GRAY2RGB,
        }

        key = (from_space.upper(), to_space.upper())
        if key in conversion_map:
            return cv2.cvtColor(frame, conversion_map[key])
        return frame

    def get_video_info(self, video_path: str) -> dict:
        """
        비디오 정보 조회

        Args:
            video_path: 비디오 파일 경로

        Returns:
            비디오 정보 딕셔너리
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return {}

        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            / max(1, cap.get(cv2.CAP_PROP_FPS)),
        }

        cap.release()
        return info
