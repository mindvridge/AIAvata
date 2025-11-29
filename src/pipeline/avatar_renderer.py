"""
Avatar Renderer using MuseTalk + Idle Loops.

MuseTalk 1.5 + Idle 루프 기반 아바타 렌더링 모듈
특징:
- 30 FPS 실시간 립싱크
- 감정 기반 idle 루프 선택
- MediaPipe 얼굴 감지 (InsightFace 대체 - 상업적 라이선스)
- MIT 라이선스 (상업적 사용 가능)
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import cv2
import numpy as np

from ..models.emotion import Emotion, EmotionMapping
from ..models.schemas import VideoFrame
from ..models.integrations import MuseTalkModel, LivePortraitModel

logger = logging.getLogger(__name__)


class AvatarRenderer:
    """
    MuseTalk 1.5 + Idle 루프 기반 아바타 렌더러

    Features:
    - 실시간 립싱크 (오디오 → 입 움직임)
    - 감정별 idle 루프 영상 관리
    - MediaPipe 얼굴 감지
    - 30 FPS 출력
    """

    def __init__(
        self,
        idle_loops_dir: str = "assets/idle_loops",
        avatar_image_path: Optional[str] = None,
        output_width: int = 512,
        output_height: int = 512,
        target_fps: int = 30,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        """
        Initialize Avatar Renderer.

        Args:
            idle_loops_dir: Idle 루프 영상 디렉토리
            avatar_image_path: 아바타 소스 이미지 경로
            output_width: 출력 비디오 너비
            output_height: 출력 비디오 높이
            target_fps: 목표 프레임 레이트
            device: Compute device
            use_fp16: FP16 추론 사용 여부
        """
        self.idle_loops_dir = Path(idle_loops_dir)
        self.avatar_image_path = avatar_image_path
        self.output_width = output_width
        self.output_height = output_height
        self.target_fps = target_fps
        self.device = device
        self.use_fp16 = use_fp16
        self.frame_duration = 1.0 / target_fps

        # 상태
        self._idle_loops: Dict[Emotion, List[np.ndarray]] = {}
        self._current_emotion = Emotion.NEUTRAL
        self._current_frame_idx = 0

        # 모델 인스턴스
        self._musetalk_model: Optional[MuseTalkModel] = None
        self._live_portrait_model: Optional[LivePortraitModel] = None
        self._face_mesh = None

        # 소스 이미지
        self._source_image: Optional[np.ndarray] = None

        self._initialized = False

    async def initialize(self) -> None:
        """모델 및 리소스 초기화"""
        if self._initialized:
            return

        logger.info("Initializing Avatar Renderer...")

        # 소스 이미지 로드
        if self.avatar_image_path:
            await self._load_source_image()

        # MediaPipe Face Mesh 초기화
        await self._init_face_mesh()

        # MuseTalk 립싱크 모델 초기화
        await self._init_lipsync_model()

        # LivePortrait 모델 초기화
        await self._init_live_portrait_model()

        # Idle 루프 로드 또는 생성
        await self._load_idle_loops()

        self._initialized = True
        logger.info("Avatar Renderer initialized successfully")

    async def _load_source_image(self) -> None:
        """소스 아바타 이미지 로드"""
        if not self.avatar_image_path:
            return

        path = Path(self.avatar_image_path)
        if not path.exists():
            logger.warning(f"Avatar image not found: {path}")
            return

        try:
            self._source_image = cv2.imread(str(path))
            if self._source_image is not None:
                self._source_image = cv2.resize(
                    self._source_image,
                    (self.output_width, self.output_height),
                )
                logger.info(f"Loaded avatar source image: {path}")
        except Exception as e:
            logger.error(f"Failed to load avatar image: {e}")

    async def _init_face_mesh(self) -> None:
        """MediaPipe Face Mesh 초기화"""
        try:
            import mediapipe as mp

            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            logger.info("MediaPipe Face Mesh initialized")

        except ImportError:
            logger.warning("MediaPipe not installed. Face detection will be limited.")

    async def _init_lipsync_model(self) -> None:
        """MuseTalk 립싱크 모델 초기화"""
        try:
            self._musetalk_model = MuseTalkModel(
                model_dir="models/musetalk/musetalkV15",
                device=self.device,
                fp16=self.use_fp16,
            )
            success = await self._musetalk_model.initialize()

            if success:
                logger.info("MuseTalk model initialized successfully")
            else:
                logger.warning("MuseTalk initialization returned False, using fallback")

        except Exception as e:
            logger.error(f"Failed to initialize MuseTalk: {e}")
            self._musetalk_model = None

    async def _init_live_portrait_model(self) -> None:
        """LivePortrait 얼굴 애니메이션 모델 초기화"""
        try:
            self._live_portrait_model = LivePortraitModel(
                model_dir="models/live_portrait",
                device=self.device,
                fp16=self.use_fp16,
            )
            success = await self._live_portrait_model.initialize()

            if success:
                logger.info("LivePortrait model initialized successfully")

                # 소스 이미지가 있으면 특징 추출
                if self._source_image is not None:
                    await self._live_portrait_model.extract_source_features(
                        self._source_image
                    )
                    logger.info("Source image features extracted")
            else:
                logger.warning("LivePortrait initialization returned False, using fallback")

        except Exception as e:
            logger.error(f"Failed to initialize LivePortrait: {e}")
            self._live_portrait_model = None

    async def _load_idle_loops(self) -> None:
        """감정별 idle 루프 영상 로드 또는 생성"""
        self.idle_loops_dir.mkdir(parents=True, exist_ok=True)

        for emotion in Emotion:
            filename = EmotionMapping.get_idle_loop_filename(emotion)
            filepath = self.idle_loops_dir / filename

            # 기존 루프 파일이 있으면 로드
            if filepath.exists():
                frames = await self._load_video_frames(str(filepath))
                if frames:
                    self._idle_loops[emotion] = frames
                    logger.info(f"Loaded idle loop: {filename} ({len(frames)} frames)")
                    continue

            # LivePortrait로 idle 루프 생성 시도
            if self._live_portrait_model and self._source_image is not None:
                frames = await self._generate_idle_loop_with_live_portrait(emotion)
                if frames:
                    self._idle_loops[emotion] = frames
                    logger.info(f"Generated idle loop for {emotion.value}: {len(frames)} frames")
                    continue

            logger.debug(f"Idle loop not available for {emotion.value}")

        # 루프가 하나도 없으면 기본 생성
        if not self._idle_loops:
            logger.warning("No idle loops available. Creating default loops.")
            self._create_default_loops()

    async def _generate_idle_loop_with_live_portrait(
        self,
        emotion: Emotion,
        num_frames: int = 60,
    ) -> Optional[List[np.ndarray]]:
        """LivePortrait로 idle 루프 생성"""
        if not self._live_portrait_model or self._source_image is None:
            return None

        try:
            # 감정에 따른 모션 강도 설정
            emotion_intensity = {
                Emotion.NEUTRAL: 0.3,
                Emotion.HAPPY: 0.5,
                Emotion.SAD: 0.2,
                Emotion.LISTENING: 0.4,
                Emotion.THINKING: 0.35,
            }
            intensity = emotion_intensity.get(emotion, 0.3)

            # LivePortrait로 프레임 시퀀스 생성
            frames = await self._live_portrait_model.generate_idle_sequence(
                emotion=emotion.value,
                num_frames=num_frames,
                motion_intensity=intensity,
            )

            return frames

        except Exception as e:
            logger.error(f"Failed to generate idle loop for {emotion.value}: {e}")
            return None

    async def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """비디오 파일을 프레임 리스트로 로드"""
        frames = []

        def load_sync():
            cap = cv2.VideoCapture(video_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # 크기 조정
                frame = cv2.resize(frame, (self.output_width, self.output_height))
                frames.append(frame)
            cap.release()
            return frames

        return await asyncio.get_event_loop().run_in_executor(None, load_sync)

    def _create_default_loops(self) -> None:
        """소스 이미지에 간단한 애니메이션 효과를 적용한 idle 루프 생성"""
        import cv2
        import math

        # 루프 설정: 60프레임 = 2초 @ 30fps
        num_frames = 60

        # 소스 이미지가 있으면 그것을 사용
        if self._source_image is not None:
            # 소스 이미지를 출력 크기에 맞게 리사이즈
            base_frame = cv2.resize(
                self._source_image,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_LINEAR,
            )
            logger.info(f"Creating animated idle loops from source image: {self.output_width}x{self.output_height}")
        else:
            # 소스 이미지가 없으면 회색 프레임
            base_frame = np.full(
                (self.output_height, self.output_width, 3),
                128,
                dtype=np.uint8,
            )
            logger.info("Using gray frame for default loops (no source image)")

        # 각 감정별로 다른 애니메이션 효과 적용
        for emotion in [Emotion.NEUTRAL, Emotion.HAPPY, Emotion.SAD, Emotion.LISTENING]:
            frames = []
            
            for i in range(num_frames):
                # 프레임 복사
                frame = base_frame.copy().astype(np.float32)
                
                # 1. 호흡 효과 (밝기 미세 변화) - 사인 곡선
                breath_factor = 1.0 + 0.02 * math.sin(2 * math.pi * i / num_frames)
                frame = frame * breath_factor
                
                # 2. 미세한 움직임 효과 (아주 작은 이동)
                shift_x = int(1.5 * math.sin(2 * math.pi * i / num_frames))
                shift_y = int(1.0 * math.sin(4 * math.pi * i / num_frames))
                
                # 이동 행렬 생성
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                frame = cv2.warpAffine(
                    frame.astype(np.uint8), 
                    M, 
                    (self.output_width, self.output_height),
                    borderMode=cv2.BORDER_REFLECT
                ).astype(np.float32)
                
                # 3. 감정별 추가 효과
                if emotion == Emotion.HAPPY:
                    # 밝게
                    frame = frame * 1.05
                elif emotion == Emotion.SAD:
                    # 약간 어둡게
                    frame = frame * 0.95
                elif emotion == Emotion.LISTENING:
                    # 약간 파란 톤 추가
                    frame[:, :, 0] = frame[:, :, 0] * 1.02  # Blue channel
                
                # 클리핑하여 0-255 범위로 유지
                frame = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(frame)
            
            self._idle_loops[emotion] = frames
            logger.info(f"Created {num_frames} animated frames for emotion: {emotion.value}")

    def _ensure_initialized(self) -> None:
        """초기화 확인"""
        if not self._initialized:
            raise RuntimeError(
                "Avatar Renderer not initialized. Call initialize() first."
            )

    def set_emotion(self, emotion: Emotion) -> None:
        """
        아바타 감정 상태 변경

        Args:
            emotion: 새로운 감정 상태
        """
        if emotion != self._current_emotion:
            logger.debug(f"Emotion changed: {self._current_emotion} → {emotion}")
            self._current_emotion = emotion
            # 프레임 인덱스는 유지하여 부드러운 전환

    def get_current_emotion(self) -> Emotion:
        """현재 감정 상태 반환"""
        return self._current_emotion

    def get_idle_frame(self) -> np.ndarray:
        """
        현재 감정의 idle 루프에서 다음 프레임 반환

        Returns:
            비디오 프레임 (BGR format)
        """
        emotion = self._current_emotion

        # 해당 감정의 루프가 없으면 neutral 사용
        if emotion not in self._idle_loops:
            emotion = Emotion.NEUTRAL

        if emotion not in self._idle_loops or len(self._idle_loops[emotion]) == 0:
            # 모든 루프가 없으면 기본 프레임 생성 (아바타 이미지 기반)
            if self._source_image is not None:
                # 소스 이미지가 있으면 그대로 반환
                return self._source_image.copy()
            else:
                # 소스 이미지도 없으면 회색 배경
                logger.warning("No idle loops available, using gray frame")
                frame = np.full(
                    (self.output_height, self.output_width, 3),
                    128,
                    dtype=np.uint8,
                )
                # 중앙에 텍스트 추가
                cv2.putText(
                    frame,
                    "Avatar",
                    (self.output_width // 2 - 50, self.output_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                )
                return frame

        frames = self._idle_loops[emotion]
        if len(frames) == 0:
            logger.warning(f"Empty idle loop for emotion: {emotion}")
            # 빈 프레임 리스트면 기본 프레임 반환
            return np.full(
                (self.output_height, self.output_width, 3),
                128,
                dtype=np.uint8,
            )
        
        frame = frames[self._current_frame_idx % len(frames)]
        self._current_frame_idx += 1

        return frame.copy()

    async def render_idle_stream(
        self,
        duration: float = -1,
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        Idle 루프 프레임 스트리밍

        Args:
            duration: 스트리밍 지속 시간 (초). -1이면 무한

        Yields:
            VideoFrame: 비디오 프레임
        """
        self._ensure_initialized()

        start_time = time.time()
        frame_index = 0

        while True:
            frame_start = time.time()

            # 현재 idle 프레임 가져오기
            frame = self.get_idle_frame()

            # JPEG 인코딩
            _, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

            yield VideoFrame(
                data=encoded.tobytes(),
                width=self.output_width,
                height=self.output_height,
                timestamp=time.time(),
                frame_index=frame_index,
                encoding="jpeg",
            )

            frame_index += 1

            # 프레임 레이트 조절
            elapsed = time.time() - frame_start
            sleep_time = self.frame_duration - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            # 지속 시간 체크
            if duration > 0 and (time.time() - start_time) >= duration:
                break

    async def render_with_audio(
        self,
        audio_stream: AsyncGenerator[bytes, None],
        audio_sample_rate: int = 24000,
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        오디오에 맞춰 립싱크된 프레임 스트리밍

        Args:
            audio_stream: 오디오 데이터 스트림
            audio_sample_rate: 오디오 샘플레이트

        Yields:
            VideoFrame: 립싱크된 비디오 프레임
        """
        self._ensure_initialized()

        frame_index = 0
        audio_buffer = b""

        # 프레임당 필요한 오디오 샘플 수
        samples_per_frame = int(audio_sample_rate / self.target_fps)
        bytes_per_frame = samples_per_frame * 2  # 16-bit audio

        async for audio_chunk in audio_stream:
            audio_buffer += audio_chunk

            # 충분한 오디오가 쌓이면 프레임 생성
            while len(audio_buffer) >= bytes_per_frame:
                frame_start = time.time()

                # 프레임에 해당하는 오디오 추출
                frame_audio = audio_buffer[:bytes_per_frame]
                audio_buffer = audio_buffer[bytes_per_frame:]

                # 현재 idle 프레임 가져오기
                base_frame = self.get_idle_frame()

                # 립싱크 적용
                lipsync_frame = await self._apply_lipsync(base_frame, frame_audio)

                # JPEG 인코딩
                _, encoded = cv2.imencode(
                    ".jpg", lipsync_frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                )

                yield VideoFrame(
                    data=encoded.tobytes(),
                    width=self.output_width,
                    height=self.output_height,
                    timestamp=time.time(),
                    frame_index=frame_index,
                    encoding="jpeg",
                )

                frame_index += 1

                # 프레임 레이트 조절
                elapsed = time.time() - frame_start
                sleep_time = self.frame_duration - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

    async def _apply_lipsync(
        self, frame: np.ndarray, audio_chunk: bytes
    ) -> np.ndarray:
        """
        립싱크 적용 (MuseTalk 또는 시뮬레이션)

        Args:
            frame: 원본 프레임
            audio_chunk: 해당 프레임의 오디오 데이터

        Returns:
            립싱크 적용된 프레임
        """
        # MuseTalk 모델이 있으면 사용
        if self._musetalk_model and hasattr(self._musetalk_model, 'process_frame'):
            try:
                # bytes를 numpy array로 변환
                if len(audio_chunk) == 0:
                    logger.debug("Empty audio chunk, skipping lip sync")
                    return frame
                    
                audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
                audio_array = audio_array / 32767.0  # Normalize to [-1, 1]

                logger.debug(f"Applying MuseTalk lip sync: frame shape={frame.shape}, audio samples={len(audio_array)}")

                # MuseTalk 추론
                lipsync_frame = await self._musetalk_model.process_frame(
                    source_frame=frame,
                    audio_chunk=audio_array,
                    audio_sample_rate=24000,
                )
                
                if lipsync_frame is not None:
                    logger.debug(f"MuseTalk lip sync successful: output shape={lipsync_frame.shape}")
                    return lipsync_frame
                else:
                    logger.warning("MuseTalk returned None, using simulation")
            except Exception as e:
                logger.error(f"MuseTalk lip sync failed: {e}", exc_info=True)

        # MuseTalk이 없거나 실패하면 간단한 시뮬레이션 사용
        logger.debug("Using lip sync simulation (MuseTalk not available or failed)")
        return await self._simulate_lipsync(frame, audio_chunk)

    async def _simulate_lipsync(
        self, frame: np.ndarray, audio_chunk: bytes
    ) -> np.ndarray:
        """
        간단한 립싱크 시뮬레이션
        오디오 레벨에 따라 입 모양을 시각적으로 변경
        """
        try:
            import cv2

            # 오디오 레벨 계산
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            audio_level = np.abs(audio_array).mean() / 32767.0  # 0.0 ~ 1.0

            # 입 열림 정도 (0 = 닫힘, 1 = 최대 열림)
            mouth_openness = min(audio_level * 3.0, 1.0)  # 레벨을 3배 증폭

            # 프레임 복사
            result_frame = frame.copy().astype(np.float32)

            # 입 영역 찾기 (대략적인 위치 - MediaPipe로 더 정확하게 할 수 있음)
            h, w = frame.shape[:2]
            mouth_y = int(h * 0.65)  # 입 위치 (얼굴 하단 65%)
            mouth_x = int(w * 0.5)   # 중심
            mouth_w = int(w * 0.15)  # 입 너비
            mouth_h = int(h * 0.08 * mouth_openness)  # 입 높이 (레벨에 따라)

            # 입 열림 시각화 (어둡게)
            if mouth_openness > 0.1:
                cv2.ellipse(
                    result_frame,
                    (mouth_x, mouth_y),
                    (mouth_w // 2, mouth_h),
                    0, 0, 360,
                    (0, 0, 0),  # 검은색
                    -1  # 채우기
                )

            # 입이 열릴 때 주변 밝기 미세 조정 (입 열림 효과)
            if mouth_openness > 0.3:
                # 입 주변을 약간 밝게
                y1 = max(0, mouth_y - mouth_h - 5)
                y2 = min(h, mouth_y + mouth_h + 5)
                x1 = max(0, mouth_x - mouth_w)
                x2 = min(w, mouth_x + mouth_w)
                result_frame[y1:y2, x1:x2] *= (1.0 + mouth_openness * 0.1)

            return np.clip(result_frame, 0, 255).astype(np.uint8)

        except Exception as e:
            logger.error(f"Lip sync simulation error: {e}")
            return frame

    def detect_face_landmarks(self, image: np.ndarray) -> Optional[dict]:
        """
        MediaPipe로 얼굴 랜드마크 감지

        Args:
            image: 입력 이미지 (BGR)

        Returns:
            랜드마크 정보 딕셔너리 또는 None
        """
        if self._face_mesh is None:
            return None

        # BGR → RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]

        # 주요 랜드마크 추출
        return {
            "landmarks": [
                {"x": lm.x * w, "y": lm.y * h, "z": lm.z}
                for lm in landmarks.landmark
            ],
            # 입 영역 (립싱크용)
            "mouth_landmarks": self._extract_mouth_landmarks(landmarks, w, h),
        }

    def _extract_mouth_landmarks(self, landmarks, width: int, height: int) -> list:
        """입 영역 랜드마크 추출"""
        # MediaPipe Face Mesh 입 랜드마크 인덱스
        mouth_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 78, 191, 80, 81, 82, 13, 312, 311, 310,
            415, 308, 324, 318, 402, 317,
        ]

        mouth_points = []
        for idx in mouth_indices:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                mouth_points.append({
                    "x": lm.x * width,
                    "y": lm.y * height,
                })

        return mouth_points

    async def cleanup(self) -> None:
        """리소스 정리"""
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None

        if self._musetalk_model:
            await self._musetalk_model.cleanup()
            self._musetalk_model = None

        if self._live_portrait_model:
            await self._live_portrait_model.cleanup()
            self._live_portrait_model = None

        self._idle_loops.clear()
        self._source_image = None
        self._initialized = False
        logger.info("Avatar Renderer cleaned up")
