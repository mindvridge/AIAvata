"""
Idle Loop Manager for avatar animations.

아바타 Idle 루프 영상 관리 서비스
특징:
- 감정별 idle 루프 영상 관리
- 부드러운 전환 효과
- 프레임 시퀀스 최적화
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..models.emotion import Emotion, EmotionMapping

logger = logging.getLogger(__name__)


class IdleLoopManager:
    """
    Idle 루프 영상 관리자

    Features:
    - 감정별 루프 영상 로드 및 관리
    - 프레임 시퀀스 접근
    - 감정 간 부드러운 전환
    - 메모리 최적화
    """

    def __init__(
        self,
        idle_loops_dir: str = "assets/idle_loops",
        target_size: Tuple[int, int] = (512, 512),
        target_fps: int = 30,
    ):
        """
        Initialize Idle Loop Manager.

        Args:
            idle_loops_dir: Idle 루프 영상 디렉토리
            target_size: 출력 프레임 크기 (width, height)
            target_fps: 목표 프레임 레이트
        """
        self.idle_loops_dir = Path(idle_loops_dir)
        self.target_size = target_size
        self.target_fps = target_fps

        # 로드된 루프
        self._loops: Dict[Emotion, List[np.ndarray]] = {}

        # 현재 상태
        self._current_emotion = Emotion.NEUTRAL
        self._current_frame_idx = 0

        # 전환 상태
        self._transition_active = False
        self._transition_progress = 0.0
        self._transition_from: Optional[Emotion] = None
        self._transition_duration_frames = 15  # 0.5초 @ 30fps

        self._initialized = False

    async def initialize(self) -> None:
        """루프 영상 로드"""
        if self._initialized:
            return

        logger.info(f"Loading idle loops from {self.idle_loops_dir}")

        if not self.idle_loops_dir.exists():
            logger.warning(f"Idle loops directory not found: {self.idle_loops_dir}")
            self._create_default_loops()
            self._initialized = True
            return

        # 모든 감정에 대해 루프 로드 시도
        load_tasks = []
        for emotion in Emotion:
            filename = EmotionMapping.get_idle_loop_filename(emotion)
            filepath = self.idle_loops_dir / filename

            if filepath.exists():
                load_tasks.append(self._load_loop_async(emotion, filepath))

        # 병렬 로드
        if load_tasks:
            await asyncio.gather(*load_tasks)

        # 기본 루프가 없으면 생성
        if Emotion.NEUTRAL not in self._loops:
            self._create_default_loops()

        self._initialized = True
        logger.info(f"Loaded {len(self._loops)} idle loops")

    async def _load_loop_async(
        self, emotion: Emotion, filepath: Path
    ) -> None:
        """비동기 루프 로드"""
        frames = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._load_video_frames(filepath)
        )
        if frames:
            self._loops[emotion] = frames
            logger.info(f"Loaded {emotion.value}: {len(frames)} frames")

    def _load_video_frames(self, video_path: Path) -> List[np.ndarray]:
        """비디오를 프레임 리스트로 로드"""
        frames = []

        try:
            cap = cv2.VideoCapture(str(video_path))

            # 원본 FPS 확인
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = max(1, int(original_fps / self.target_fps))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 프레임 스킵 (FPS 맞춤)
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue

                # 크기 조정
                frame = cv2.resize(
                    frame, self.target_size, interpolation=cv2.INTER_LINEAR
                )
                frames.append(frame)
                frame_count += 1

            cap.release()

        except Exception as e:
            logger.error(f"Failed to load video {video_path}: {e}")

        return frames

    def _create_default_loops(self) -> None:
        """기본 프레임으로 루프 생성"""
        logger.info("Creating default idle loops")

        # 회색 배경 기본 프레임
        default_frame = np.full(
            (self.target_size[1], self.target_size[0], 3),
            200,
            dtype=np.uint8,
        )

        # 약간의 움직임 효과 추가 (밝기 변화)
        frames = []
        for i in range(self.target_fps):  # 1초 루프
            # 부드러운 밝기 변화
            brightness = 200 + int(10 * np.sin(2 * np.pi * i / self.target_fps))
            frame = np.full(
                (self.target_size[1], self.target_size[0], 3),
                brightness,
                dtype=np.uint8,
            )
            frames.append(frame)

        # 기본 감정들에 할당
        for emotion in [Emotion.NEUTRAL, Emotion.LISTENING, Emotion.THINKING]:
            self._loops[emotion] = frames.copy()

    def get_frame(self, emotion: Optional[Emotion] = None) -> np.ndarray:
        """
        현재 프레임 반환

        Args:
            emotion: 감정 (없으면 현재 감정 사용)

        Returns:
            비디오 프레임 (BGR)
        """
        target_emotion = emotion or self._current_emotion

        # 전환 중이면 블렌딩된 프레임 반환
        if self._transition_active and self._transition_from:
            return self._get_transition_frame()

        return self._get_emotion_frame(target_emotion)

    def _get_emotion_frame(self, emotion: Emotion) -> np.ndarray:
        """특정 감정의 프레임 반환"""
        # 해당 감정 루프가 없으면 neutral 사용
        if emotion not in self._loops:
            emotion = Emotion.NEUTRAL

        # 여전히 없으면 기본 프레임
        if emotion not in self._loops:
            return np.full(
                (self.target_size[1], self.target_size[0], 3),
                200,
                dtype=np.uint8,
            )

        frames = self._loops[emotion]
        frame = frames[self._current_frame_idx % len(frames)]
        return frame.copy()

    def _get_transition_frame(self) -> np.ndarray:
        """전환 중 블렌딩된 프레임 반환"""
        from_frame = self._get_emotion_frame(self._transition_from)
        to_frame = self._get_emotion_frame(self._current_emotion)

        # 선형 블렌딩
        alpha = self._transition_progress
        blended = cv2.addWeighted(
            from_frame, 1 - alpha, to_frame, alpha, 0
        )

        return blended

    def advance_frame(self) -> None:
        """다음 프레임으로 이동"""
        self._current_frame_idx += 1

        # 전환 진행
        if self._transition_active:
            self._transition_progress += 1.0 / self._transition_duration_frames
            if self._transition_progress >= 1.0:
                self._transition_active = False
                self._transition_from = None
                self._transition_progress = 0.0

    def set_emotion(
        self,
        emotion: Emotion,
        smooth_transition: bool = True,
    ) -> None:
        """
        감정 상태 변경

        Args:
            emotion: 새로운 감정
            smooth_transition: 부드러운 전환 사용 여부
        """
        if emotion == self._current_emotion:
            return

        if smooth_transition and not self._transition_active:
            self._transition_active = True
            self._transition_from = self._current_emotion
            self._transition_progress = 0.0

        self._current_emotion = emotion
        logger.debug(f"Emotion changed to: {emotion.value}")

    def get_current_emotion(self) -> Emotion:
        """현재 감정 반환"""
        return self._current_emotion

    def get_available_emotions(self) -> List[Emotion]:
        """사용 가능한 감정 목록 반환"""
        return list(self._loops.keys())

    def get_loop_info(self, emotion: Emotion) -> Optional[Dict]:
        """루프 정보 반환"""
        if emotion not in self._loops:
            return None

        frames = self._loops[emotion]
        return {
            "emotion": emotion.value,
            "frame_count": len(frames),
            "duration_seconds": len(frames) / self.target_fps,
            "frame_size": frames[0].shape if frames else None,
        }

    async def add_loop(
        self,
        emotion: Emotion,
        video_path: str,
    ) -> bool:
        """
        새 루프 추가

        Args:
            emotion: 감정
            video_path: 비디오 파일 경로

        Returns:
            성공 여부
        """
        path = Path(video_path)
        if not path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False

        frames = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._load_video_frames(path)
        )

        if not frames:
            return False

        self._loops[emotion] = frames
        logger.info(f"Added loop for {emotion.value}: {len(frames)} frames")
        return True

    def remove_loop(self, emotion: Emotion) -> bool:
        """루프 제거"""
        if emotion in self._loops:
            del self._loops[emotion]
            logger.info(f"Removed loop for {emotion.value}")
            return True
        return False

    async def cleanup(self) -> None:
        """리소스 정리"""
        self._loops.clear()
        self._initialized = False
        logger.info("Idle Loop Manager cleaned up")
