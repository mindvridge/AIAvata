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
from typing import Any, AsyncGenerator, Dict, List, Optional

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
        driving_video_path: Optional[str] = None,
        output_width: int = 512,
        output_height: int = 512,
        target_fps: int = 30,
        device: str = "cuda",
        use_fp16: bool = True,
        settings: Optional[Any] = None,
    ):
        """
        Initialize Avatar Renderer.

        Args:
            idle_loops_dir: Idle 루프 영상 디렉토리
            avatar_image_path: 아바타 소스 이미지 경로
            driving_video_path: 드라이빙 비디오 경로 (idle 애니메이션용)
            output_width: 출력 비디오 너비
            output_height: 출력 비디오 높이
            target_fps: 목표 프레임 레이트
            device: Compute device
            use_fp16: FP16 추론 사용 여부
            settings: Application settings
        """
        self.idle_loops_dir = Path(idle_loops_dir)
        self.avatar_image_path = avatar_image_path
        self.driving_video_path = driving_video_path
        self.output_width = output_width
        self.output_height = output_height
        self.target_fps = target_fps
        self.device = device
        self.use_fp16 = use_fp16
        self.frame_duration = 1.0 / target_fps
        self._settings = settings

        # 상태
        self._idle_loops: Dict[Emotion, List[np.ndarray]] = {}
        # 비디오 파일 직접 스트리밍용 (메모리 효율적)
        self._idle_video_paths: Dict[Emotion, Optional[Path]] = {}
        # 프레임 캐싱 설정
        self._cache_enabled = getattr(settings, 'idle_loop_cache_enabled', True) if settings else True
        self._cache_max_frames = getattr(settings, 'idle_loop_cache_max_frames', 300) if settings else 300
        self._cache_max_duration = getattr(settings, 'idle_loop_cache_max_duration_seconds', 10.0) if settings else 10.0
        # 캐시된 비디오 정보 (비디오가 캐시되었는지 여부)
        self._cached_videos: Dict[Emotion, bool] = {}
        self._current_emotion = Emotion.NEUTRAL
        self._current_frame_idx = 0
        self._loop_length = 0  # 현재 루프의 총 프레임 수
        self._waiting_for_loop_end = False  # 루프 끝 대기 플래그
        self._loop_end_event: Optional[asyncio.Event] = None  # 루프 끝 이벤트

        # 모델 인스턴스
        self._musetalk_model: Optional[MuseTalkModel] = None
        self._live_portrait_model: Optional[LivePortraitModel] = None
        self._face_mesh = None
        self._face_landmarker = None  # tasks API용
        self._use_tasks_api = False  # tasks API 사용 여부

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
        """MediaPipe Face Mesh 초기화 (tasks API 사용)"""
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options
            from mediapipe.tasks.python.vision.core import vision_task_running_mode

            # MediaPipe 0.10.0+ tasks API 사용
            if hasattr(mp, 'solutions'):
                # 구버전 solutions API 사용
                self._mp_face_mesh = mp.solutions.face_mesh
                self._face_mesh = self._mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self._use_tasks_api = False
                logger.info("MediaPipe Face Mesh initialized (solutions API)")
            else:
                # 새로운 tasks API 사용
                try:
                    base_opts = base_options.BaseOptions(
                        model_asset_path=None,  # 번들된 모델 사용
                        delegate=base_options.BaseOptions.Delegate.CPU
                    )
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_opts,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                        num_faces=1,
                        min_face_detection_confidence=0.5,
                        min_face_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                        running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO
                    )
                    self._face_landmarker = vision.FaceLandmarker.create_from_options(options)
                    self._use_tasks_api = True
                    logger.info("MediaPipe Face Landmarker initialized (tasks API)")
                except Exception as e:
                    logger.warning(f"Failed to initialize MediaPipe tasks API: {e}. Using fallback.")
                    self._face_landmarker = None
                    self._use_tasks_api = False

        except (ImportError, AttributeError) as e:
            logger.warning(f"MediaPipe not available: {e}. Face detection will be limited.")
            self._face_mesh = None
            self._face_landmarker = None
            self._use_tasks_api = False

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
                driving_video_path=self.driving_video_path,
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

                # 드라이빙 비디오 로드 (있으면)
                if self.driving_video_path:
                    await self._live_portrait_model.load_driving_video(self.driving_video_path)
                else:
                    # 기본 드라이빙 비디오 사용 시도
                    default_driving_paths = [
                        "assets/avatars/avata_ani.mp4",
                        "external/LivePortrait/assets/examples/driving/d0.mp4",
                    ]
                    for path in default_driving_paths:
                        if Path(path).exists():
                            logger.info(f"Using default driving video: {path}")
                            await self._live_portrait_model.load_driving_video(path)
                            break
            else:
                logger.warning("LivePortrait initialization returned False, using fallback")

        except Exception as e:
            logger.error(f"Failed to initialize LivePortrait: {e}")
            self._live_portrait_model = None

    async def _load_idle_loops(self) -> None:
        """감정별 idle 루프 영상 로드 또는 생성"""
        self.idle_loops_dir.mkdir(parents=True, exist_ok=True)

        # 🎬 우선순위 1: Pre-rendered idle loop 비디오 직접 로드 (최고 품질)
        # avata_ani.mp4를 기본 루프 영상으로 사용
        prerendered_paths = [
            Path("assets/avatars/avata_ani.mp4"),  # 기본 루프 영상
        ]

        for prerendered_path in prerendered_paths:
            if prerendered_path.exists():
                video_path = Path(prerendered_path)
                
                # 비디오 정보 확인 (프레임 수, 길이)
                video_info = await self._get_video_info(str(video_path))
                total_frames = video_info.get("total_frames", 0)
                duration = video_info.get("duration", 0.0)
                fps = video_info.get("fps", self.target_fps)
                
                # 캐싱 여부 결정
                should_cache = (
                    self._cache_enabled and
                    total_frames > 0 and
                    total_frames <= self._cache_max_frames and
                    duration <= self._cache_max_duration
                )
                
                if should_cache:
                    # 🎬 프레임 캐싱 모드: 짧은 비디오는 메모리에 로드
                    logger.info(f"✅ Pre-rendered idle loop 등록: {prerendered_path}")
                    logger.info(f"   프레임 캐싱 모드: {total_frames} 프레임 ({duration:.1f}초) → 메모리 로드")
                    
                    frames = await self._load_video_frames(str(prerendered_path))
                    if frames:
                        self._idle_loops[Emotion.NEUTRAL] = frames
                        self._idle_loops[Emotion.HAPPY] = frames
                        self._idle_loops[Emotion.SAD] = frames
                        self._idle_loops[Emotion.LISTENING] = frames
                        self._cached_videos[Emotion.NEUTRAL] = True
                        self._cached_videos[Emotion.HAPPY] = True
                        self._cached_videos[Emotion.SAD] = True
                        self._cached_videos[Emotion.LISTENING] = True
                        logger.info(f"   ✅ {len(frames)} 프레임 메모리 캐시 완료")
                        logger.info(f"   Resolution: {self.output_width}x{self.output_height}, FPS target: {self.target_fps}")
                else:
                    # 🎬 파일 스트리밍 모드: 긴 비디오는 파일에서 직접 읽기
                    self._idle_video_paths[Emotion.NEUTRAL] = video_path
                    self._idle_video_paths[Emotion.HAPPY] = video_path
                    self._idle_video_paths[Emotion.SAD] = video_path
                    self._idle_video_paths[Emotion.LISTENING] = video_path
                    self._cached_videos[Emotion.NEUTRAL] = False
                    self._cached_videos[Emotion.HAPPY] = False
                    self._cached_videos[Emotion.SAD] = False
                    self._cached_videos[Emotion.LISTENING] = False
                    logger.info(f"✅ Pre-rendered idle loop 등록: {prerendered_path}")
                    logger.info(f"   파일 스트리밍 모드: {total_frames} 프레임 ({duration:.1f}초) → 파일에서 직접 읽기")
                    logger.info(f"   Resolution: {self.output_width}x{self.output_height}, FPS target: {self.target_fps}")
                
                return  # Pre-rendered 영상 등록 완료

        # avata_ani.mp4가 없으면 경고
        logger.warning(f"⚠️ avata_ani.mp4를 찾을 수 없습니다: assets/avatars/avata_ani.mp4")
        logger.warning("   idle 루프 영상이 없어 idle 상태에서 기본 이미지를 사용합니다.")

    async def _generate_idle_loop_with_live_portrait(
        self,
        emotion: Emotion,
        num_frames: int = 60,
    ) -> Optional[List[np.ndarray]]:
        """LivePortrait로 idle 루프 생성"""
        if not self._live_portrait_model or self._source_image is None:
            return None

        try:
            # num_frames를 duration_seconds로 변환
            duration_seconds = num_frames / self.target_fps

            # LivePortrait로 프레임 시퀀스 생성
            frames = await self._live_portrait_model.generate_idle_sequence(
                source_image=self._source_image,
                emotion=emotion.value,
                duration_seconds=duration_seconds,
                fps=self.target_fps,
            )

            return frames

        except Exception as e:
            logger.error(f"Failed to generate idle loop for {emotion.value}: {e}")
            return None

    async def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """비디오 파일 정보 가져오기"""
        def get_info():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"total_frames": 0, "fps": 0, "duration": 0.0}
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
            duration = total_frames / fps if fps > 0 else 0.0
            
            cap.release()
            return {
                "total_frames": total_frames,
                "fps": fps,
                "duration": duration,
            }
        
        return await asyncio.get_event_loop().run_in_executor(None, get_info)

    async def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """비디오 파일을 프레임 리스트로 로드 (캐싱용)"""
        frames = []

        def load_sync():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return []
            
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

                # 1. 호흡 효과 (밝기 변화) - 사인 곡선 (눈에 보이도록 증가)
                breath_factor = 1.0 + 0.05 * math.sin(2 * math.pi * i / num_frames)
                frame = frame * breath_factor

                # 2. 미세한 움직임 효과 (자연스러운 흔들림)
                shift_x = int(3 * math.sin(2 * math.pi * i / num_frames))
                shift_y = int(2 * math.sin(4 * math.pi * i / num_frames))

                # 3. 미세한 회전 효과 추가 (호흡처럼 보이게)
                rotation_angle = 0.5 * math.sin(2 * math.pi * i / num_frames)
                center = (self.output_width // 2, self.output_height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                rotation_matrix[0, 2] += shift_x
                rotation_matrix[1, 2] += shift_y

                frame = cv2.warpAffine(
                    frame.astype(np.uint8),
                    rotation_matrix,
                    (self.output_width, self.output_height),
                    borderMode=cv2.BORDER_REFLECT
                ).astype(np.float32)

                # 4. 감정별 추가 효과
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

    def _reset_lipsync_buffer(self) -> None:
        """
        립싱크 오디오 버퍼 초기화

        새 문장/세션 시작 전 호출하여 이전 오디오 데이터가
        새 립싱크에 영향을 주지 않도록 함
        """
        if self._musetalk_model and hasattr(self._musetalk_model, 'reset_audio_buffer'):
            self._musetalk_model.reset_audio_buffer()
            logger.debug("MuseTalk audio buffer reset")

    def get_loop_progress(self) -> tuple[int, int]:
        """
        현재 루프 진행 상태 반환

        Returns:
            (현재 프레임 인덱스, 총 프레임 수)
        """
        return (self._current_frame_idx % max(1, self._loop_length), self._loop_length)

    def is_at_loop_start(self) -> bool:
        """루프 시작 지점인지 확인 (자연스러운 전환 포인트)"""
        if self._loop_length == 0:
            return True
        return (self._current_frame_idx % self._loop_length) == 0

    async def wait_for_loop_end(self, timeout: float = 5.0) -> bool:
        """
        현재 루프가 끝날 때까지 대기

        Args:
            timeout: 최대 대기 시간 (초)

        Returns:
            루프 완료 여부 (timeout 시 False)
        """
        if self._loop_length == 0:
            return True

        # 이벤트 생성 및 대기 플래그 설정
        self._loop_end_event = asyncio.Event()
        self._waiting_for_loop_end = True

        try:
            await asyncio.wait_for(self._loop_end_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Loop end wait timeout ({timeout}s)")
            return False
        finally:
            self._waiting_for_loop_end = False
            self._loop_end_event = None

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
        self._loop_length = len(frames)  # 루프 길이 업데이트

        if len(frames) == 0:
            logger.warning(f"Empty idle loop for emotion: {emotion}")
            # 빈 프레임 리스트면 기본 프레임 반환
            return np.full(
                (self.output_height, self.output_width, 3),
                128,
                dtype=np.uint8,
            )

        # 현재 프레임 인덱스 (루프 내)
        frame_idx_in_loop = self._current_frame_idx % len(frames)
        frame = frames[frame_idx_in_loop]
        self._current_frame_idx += 1

        # 루프 끝에 도달했고, 대기 중이면 이벤트 발생
        if self._waiting_for_loop_end and self._loop_end_event:
            # 다음 프레임이 루프 시작점이면 (현재가 마지막 프레임)
            next_idx = self._current_frame_idx % len(frames)
            if next_idx == 0:
                logger.debug(f"Loop end reached at frame {frame_idx_in_loop}")
                self._loop_end_event.set()

        return frame.copy()

    async def render_idle_stream(
        self,
        duration: float = -1,
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        Idle 루프 프레임 스트리밍 (하이브리드: 캐시 또는 파일 스트리밍)

        짧은 비디오는 메모리에 캐시된 프레임을 사용하고,
        긴 비디오는 파일에서 직접 스트리밍합니다.

        Args:
            duration: 스트리밍 지속 시간 (초). -1이면 무한

        Yields:
            VideoFrame: 비디오 프레임
        """
        self._ensure_initialized()

        emotion = self._current_emotion
        
        # 🎬 우선순위 1: 캐시된 프레임 사용 (가장 빠름)
        if emotion in self._idle_loops and len(self._idle_loops[emotion]) > 0:
            logger.debug(f"캐시된 프레임 사용: {emotion.value} ({len(self._idle_loops[emotion])} 프레임)")
            async for frame in self._render_idle_stream_from_memory(duration):
                yield frame
            return
        
        # 🎬 우선순위 2: 파일 스트리밍
        video_path = self._idle_video_paths.get(emotion)
        if video_path is not None and video_path.exists():
            async for frame in self._render_idle_stream_from_file(video_path, duration):
                yield frame
            return
        
        # 🎬 우선순위 3: Fallback (메모리 프레임)
        logger.debug(f"비디오 파일 없음, 메모리 프레임 방식 사용: {emotion.value}")
        async for frame in self._render_idle_stream_from_memory(duration):
            yield frame

    async def _render_idle_stream_from_file(
        self,
        video_path: Path,
        duration: float = -1,
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        파일에서 직접 스트리밍 (긴 비디오용)
        """

        logger.debug(f"🎬 파일 스트리밍 시작: {video_path}")
        
        start_time = time.time()
        frame_index = 0
        cap = None

        try:
            # 비디오 캡처 열기
            def open_video():
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
                return cap

            cap = await asyncio.get_event_loop().run_in_executor(None, open_video)
            
            # 비디오 정보 가져오기
            fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.debug(f"   파일 스트리밍: {total_frames} 프레임, {fps:.2f} FPS")

            frame_duration = 1.0 / self.target_fps
            loop_count = 0

            while True:
                frame_start = time.time()

                # 프레임 읽기
                def read_frame():
                    ret, frame = cap.read()
                    if not ret:
                        # 비디오 끝에 도달하면 처음으로 돌아가기 (루프)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if ret:
                            loop_count += 1
                            if loop_count % 10 == 0:
                                logger.debug(f"비디오 루프 #{loop_count}")
                    return ret, frame

                ret, frame = await asyncio.get_event_loop().run_in_executor(None, read_frame)
                
                if not ret:
                    logger.warning("비디오 프레임을 읽을 수 없습니다")
                    break

                # 크기 조정
                if frame.shape[1] != self.output_width or frame.shape[0] != self.output_height:
                    frame = cv2.resize(frame, (self.output_width, self.output_height))

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
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                # 지속 시간 체크
                if duration > 0 and (time.time() - start_time) >= duration:
                    break

        except Exception as e:
            logger.error(f"비디오 스트리밍 오류: {e}", exc_info=True)
            # 오류 발생 시 메모리 프레임 방식으로 fallback
            logger.info("메모리 프레임 방식으로 fallback")
            async for frame in self._render_idle_stream_from_memory(duration):
                yield frame

        finally:
            if cap is not None:
                def close_video():
                    cap.release()
                await asyncio.get_event_loop().run_in_executor(None, close_video)

    async def _render_idle_stream_from_memory(
        self,
        duration: float = -1,
    ) -> AsyncGenerator[VideoFrame, None]:
        """
        메모리에 로드된 프레임으로 idle 스트리밍 (fallback)
        """
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

        # 새 오디오 스트림 시작 전 MuseTalk 버퍼 초기화
        self._reset_lipsync_buffer()

        frame_index = 0
        audio_buffer = b""

        # 프레임당 필요한 오디오 샘플 수
        samples_per_frame = int(audio_sample_rate / self.target_fps)
        bytes_per_frame = samples_per_frame * 2  # 16-bit audio

        logger.info(f"🎬 Starting audio stream rendering: sample_rate={audio_sample_rate}, fps={self.target_fps}, bytes_per_frame={bytes_per_frame}")

        async for audio_chunk in audio_stream:
            logger.debug(f"Received audio chunk: {len(audio_chunk)} bytes, buffer: {len(audio_buffer)} bytes")
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
                lipsync_frame = await self._apply_lipsync(
                    base_frame, frame_audio, audio_sample_rate
                )

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
                if frame_index % 10 == 0:
                    logger.info(f"🎬 Generated {frame_index} lip sync frames")

                # 프레임 레이트 조절
                elapsed = time.time() - frame_start
                sleep_time = self.frame_duration - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        # 남은 버퍼 처리
        if len(audio_buffer) > 0:
            logger.debug(f"Remaining audio buffer: {len(audio_buffer)} bytes (not enough for frame)")

        logger.info(f"✅ Lip sync rendering complete: total {frame_index} frames generated")

    async def _apply_lipsync(
        self, frame: np.ndarray, audio_chunk: bytes, audio_sample_rate: int = 24000
    ) -> np.ndarray:
        """
        립싱크 적용 (MuseTalk 사용)

        Args:
            frame: 원본 프레임
            audio_chunk: 해당 프레임의 오디오 데이터 (16-bit PCM)
            audio_sample_rate: 오디오 샘플레이트

        Returns:
            립싱크 적용된 프레임
        """
        if len(audio_chunk) == 0:
            logger.debug("Empty audio chunk, skipping lip sync")
            return frame

        # fast_lipsync 설정 확인 - 빠른 시뮬레이션 사용
        if hasattr(self, '_settings') and self._settings and getattr(self._settings, 'fast_lipsync', False):
            logger.warning("⚠️ fast_lipsync=True 설정됨 - MuseTalk 비활성화 상태")
            return frame

        # MuseTalk 모델 상태 확인 및 상세 로깅
        if not self._musetalk_model:
            logger.error("❌ MuseTalk 모델이 초기화되지 않았습니다!")
            logger.error("   해결방법: run.bat를 다시 실행하여 MuseTalk 모델을 다운로드하세요.")
            return frame

        if not hasattr(self._musetalk_model, 'process_frame'):
            logger.error("❌ MuseTalk 모델에 process_frame 메서드가 없습니다!")
            logger.error("   해결방법: MuseTalk 패키지를 재설치하세요.")
            return frame

        # MuseTalk 내부 상태 확인
        if hasattr(self._musetalk_model, '_unet') and self._musetalk_model._unet is None:
            logger.error("❌ MuseTalk UNet 모델이 로드되지 않았습니다!")
            logger.error("   필요한 파일: models/musetalk/musetalkV15/unet.pth")
            logger.error("   해결방법: run.bat를 다시 실행하거나 수동으로 다운로드하세요.")
            return frame

        if hasattr(self._musetalk_model, '_vae') and self._musetalk_model._vae is None:
            logger.error("❌ MuseTalk VAE 모델이 로드되지 않았습니다!")
            logger.error("   필요한 파일: models/musetalk/sd-vae-ft-mse/")
            logger.error("   해결방법: run.bat를 다시 실행하거나 수동으로 다운로드하세요.")
            return frame

        try:
            # bytes를 numpy array로 변환
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32767.0  # Normalize to [-1, 1]

            # MuseTalk은 16kHz를 기대하므로 필요시 리샘플링
            target_sample_rate = 16000
            if audio_sample_rate != target_sample_rate:
                audio_array = self._resample_audio(
                    audio_array, audio_sample_rate, target_sample_rate
                )

            logger.debug(f"Applying MuseTalk lip sync: frame shape={frame.shape}, audio samples={len(audio_array)}")

            # MuseTalk 추론 (16kHz로 통일)
            lipsync_frame = await self._musetalk_model.process_frame(
                source_frame=frame,
                audio_chunk=audio_array,
                audio_sample_rate=target_sample_rate,
            )

            if lipsync_frame is not None and lipsync_frame.shape == frame.shape:
                logger.debug(f"MuseTalk lip sync successful: output shape={lipsync_frame.shape}")
                return lipsync_frame
            else:
                if lipsync_frame is None:
                    logger.error("❌ MuseTalk이 None을 반환했습니다. 내부 처리 오류입니다.")
                else:
                    logger.error(f"❌ MuseTalk 출력 shape 불일치: 예상={frame.shape}, 실제={lipsync_frame.shape}")
                return frame

        except Exception as e:
            logger.error(f"❌ MuseTalk 립싱크 실패: {e}", exc_info=True)
            return frame

    def _resample_audio(
        self, audio: np.ndarray, orig_sr: int, target_sr: int
    ) -> np.ndarray:
        """
        오디오 리샘플링

        Args:
            audio: 오디오 데이터 (float32)
            orig_sr: 원본 샘플레이트
            target_sr: 목표 샘플레이트

        Returns:
            리샘플링된 오디오
        """
        if orig_sr == target_sr:
            return audio

        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # librosa가 없으면 간단한 선형 보간
            ratio = target_sr / orig_sr
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    async def _simulate_lipsync(
        self, frame: np.ndarray, audio_chunk: bytes
    ) -> np.ndarray:
        """
        립싱크 시뮬레이션 (MuseTalk 대체)

        오디오 레벨에 따라 입 부분에 눈에 보이는 변화 적용
        """
        try:
            # 오디오 레벨 계산
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            if len(audio_array) == 0:
                return frame

            # 에너지 레벨 계산 (0.0 ~ 1.0)
            energy = np.sqrt(np.mean(audio_array ** 2)) / 32767.0

            # 너무 낮은 에너지면 처리하지 않음
            if energy < 0.01:
                return frame

            result = frame.copy()
            h, w = frame.shape[:2]

            # 입 영역 (더 넓은 영역)
            mouth_top = int(h * 0.55)
            mouth_bottom = int(h * 0.85)
            mouth_left = int(w * 0.25)
            mouth_right = int(w * 0.75)

            # 입 영역에 밝기 변화 적용
            mouth_region = result[mouth_top:mouth_bottom, mouth_left:mouth_right].astype(np.float32)

            # 오디오 에너지에 따른 밝기 변화 (최대 20% 밝게) - 효과 증가
            brightness_factor = 1.0 + (energy * 0.2)
            mouth_region = mouth_region * brightness_factor

            # 턱 영역에 미세한 확대 효과 (입 벌림 시뮬레이션)
            scale_factor = 1.0 + energy * 0.03
            if scale_factor != 1.0:
                mouth_h, mouth_w = mouth_region.shape[:2]
                new_h = int(mouth_h * scale_factor)
                new_w = int(mouth_w * scale_factor)
                if new_h > 0 and new_w > 0:
                    scaled = cv2.resize(mouth_region.astype(np.uint8), (new_w, new_h))
                    # 중앙 크롭
                    start_y = (new_h - mouth_h) // 2
                    start_x = (new_w - mouth_w) // 2
                    if start_y >= 0 and start_x >= 0:
                        mouth_region = scaled[start_y:start_y+mouth_h, start_x:start_x+mouth_w].astype(np.float32)

            # 클리핑
            result[mouth_top:mouth_bottom, mouth_left:mouth_right] = np.clip(
                mouth_region, 0, 255
            ).astype(np.uint8)

            return result

        except Exception as e:
            logger.error(f"Lip sync simulation error: {e}")
            return frame

    def _detect_mouth_region(
        self, frame: np.ndarray
    ) -> tuple[tuple[int, int] | None, int, int]:
        """
        MediaPipe로 입 영역 감지

        Returns:
            (mouth_center, mouth_width, mouth_height) 또는 (None, 0, 0)
        """
        import mediapipe as mp
        from mediapipe.tasks.python.vision.core import image as mp_image

        if self._use_tasks_api and self._face_landmarker is not None:
            # tasks API 사용
            try:
                h, w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
                
                import time
                timestamp_ms = int(time.time() * 1000)
                detection_result = self._face_landmarker.detect_for_video(mp_img, timestamp_ms)

                if not detection_result.face_landmarks:
                    return None, 0, 0

                landmarks = detection_result.face_landmarks[0]

                # 입 랜드마크 인덱스 (MediaPipe Face Landmarker)
                # 13: 윗입술 중앙, 14: 아랫입술 중앙
                # 78: 왼쪽 입꼬리, 308: 오른쪽 입꼬리
                if len(landmarks) > 308:
                    upper_lip = landmarks[13]
                    lower_lip = landmarks[14]
                    left_corner = landmarks[78]
                    right_corner = landmarks[308]

                    # 입 중심 계산
                    mouth_center_x = int((left_corner.x + right_corner.x) / 2 * w)
                    mouth_center_y = int((upper_lip.y + lower_lip.y) / 2 * h)

                    # 입 크기 계산
                    mouth_width = int(abs(right_corner.x - left_corner.x) * w)
                    mouth_height = int(abs(lower_lip.y - upper_lip.y) * h)

                    return (mouth_center_x, mouth_center_y), mouth_width, max(mouth_height, 5)
                else:
                    return None, 0, 0

            except Exception as e:
                logger.debug(f"Mouth detection (tasks API) failed: {e}")
                return None, 0, 0
        elif self._face_mesh is not None:
            # solutions API 사용
            try:
                h, w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._face_mesh.process(rgb_frame)

                if not results.multi_face_landmarks:
                    return None, 0, 0

                landmarks = results.multi_face_landmarks[0].landmark

                # 입 랜드마크 인덱스 (MediaPipe Face Mesh)
                # 13: 윗입술 중앙, 14: 아랫입술 중앙
                # 78: 왼쪽 입꼬리, 308: 오른쪽 입꼬리
                upper_lip = landmarks[13]
                lower_lip = landmarks[14]
                left_corner = landmarks[78]
                right_corner = landmarks[308]

                # 입 중심 계산
                mouth_center_x = int((left_corner.x + right_corner.x) / 2 * w)
                mouth_center_y = int((upper_lip.y + lower_lip.y) / 2 * h)

                # 입 크기 계산
                mouth_width = int(abs(right_corner.x - left_corner.x) * w)
                mouth_height = int(abs(lower_lip.y - upper_lip.y) * h)

                return (mouth_center_x, mouth_center_y), mouth_width, max(mouth_height, 5)

            except Exception as e:
                logger.debug(f"Mouth detection (solutions API) failed: {e}")
                return None, 0, 0
        else:
            return None, 0, 0

    def detect_face_landmarks(self, image: np.ndarray) -> Optional[dict]:
        """
        MediaPipe로 얼굴 랜드마크 감지 (solutions API 또는 tasks API)

        Args:
            image: 입력 이미지 (BGR)

        Returns:
            랜드마크 정보 딕셔너리 또는 None
        """
        import mediapipe as mp
        from mediapipe.tasks.python.vision.core import image as mp_image

        if self._use_tasks_api and self._face_landmarker is not None:
            # tasks API 사용
            try:
                # BGR → RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_image)
                
                # 비디오 모드에서는 timestamp 필요
                import time
                timestamp_ms = int(time.time() * 1000)
                detection_result = self._face_landmarker.detect_for_video(mp_img, timestamp_ms)

                if not detection_result.face_landmarks:
                    return None

                landmarks = detection_result.face_landmarks[0]
                h, w = image.shape[:2]

                # tasks API의 랜드마크는 NormalizedLandmark 리스트
                return {
                    "landmarks": [
                        {"x": lm.x * w, "y": lm.y * h, "z": lm.z * w}  # z는 깊이 (스케일 조정)
                        for lm in landmarks
                    ],
                    "mouth_landmarks": self._extract_mouth_landmarks_from_list(landmarks, w, h),
                }
            except Exception as e:
                logger.debug(f"Face landmark detection (tasks API) failed: {e}")
                return None
        elif self._face_mesh is not None:
            # solutions API 사용
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
        else:
            return None

    def _extract_mouth_landmarks(self, landmarks, width: int, height: int) -> list:
        """입 영역 랜드마크 추출 (solutions API용)"""
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

    def _extract_mouth_landmarks_from_list(self, landmarks_list, width: int, height: int) -> list:
        """입 영역 랜드마크 추출 (tasks API용)"""
        # MediaPipe Face Landmarker 입 랜드마크 인덱스 (동일)
        mouth_indices = [
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
            95, 78, 191, 80, 81, 82, 13, 312, 311, 310,
            415, 308, 324, 318, 402, 317,
        ]

        mouth_points = []
        for idx in mouth_indices:
            if idx < len(landmarks_list):
                lm = landmarks_list[idx]
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
        if self._face_landmarker:
            self._face_landmarker.close()
            self._face_landmarker = None
        self._use_tasks_api = False

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
