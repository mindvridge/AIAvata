"""
LivePortrait Integration Module.

얼굴 애니메이션 및 Idle 루프 생성
GitHub: https://github.com/KwaiVGI/LivePortrait
라이선스: MIT (상업적 사용 가능)

특징:
- 단일 이미지에서 비디오 생성
- 표정 및 포즈 제어
- 실시간 애니메이션
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# LivePortrait 모델 경로
LIVE_PORTRAIT_MODEL_DIR = os.getenv("LIVE_PORTRAIT_MODEL_DIR", "models/live_portrait")


class LivePortraitModel:
    """
    LivePortrait 얼굴 애니메이션 모델

    단일 이미지에서 다양한 표정과 움직임을 가진 비디오를 생성합니다.
    """

    def __init__(
        self,
        model_dir: str = LIVE_PORTRAIT_MODEL_DIR,
        device: str = "cuda",
        fp16: bool = True,
    ):
        """
        Initialize LivePortrait Model.

        Args:
            model_dir: 모델 파일 디렉토리
            device: 연산 디바이스
            fp16: FP16 추론 사용 여부
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.fp16 = fp16

        # 모델 컴포넌트
        self._appearance_extractor = None
        self._motion_extractor = None
        self._warping_module = None
        self._spade_generator = None

        # 소스 이미지 캐시
        self._source_cache: Dict[str, Dict[str, Any]] = {}

        self._initialized = False

    async def initialize(self) -> bool:
        """
        모델 초기화 및 로드

        Returns:
            초기화 성공 여부
        """
        if self._initialized:
            return True

        logger.info("Initializing LivePortrait model...")

        try:
            import torch

            # 모델 디렉토리 확인
            if not self.model_dir.exists():
                logger.warning(f"LivePortrait model directory not found: {self.model_dir}")
                logger.info("Run 'python tools/setup_models.py' to download models.")
                self.model_dir.mkdir(parents=True, exist_ok=True)
                self._initialized = True
                return True

            # LivePortrait 모델 로드 시도
            try:
                # 실제 LivePortrait 패키지 임포트
                from live_portrait.modules.appearance_feature_extractor import (
                    AppearanceFeatureExtractor
                )
                from live_portrait.modules.motion_extractor import MotionExtractor
                from live_portrait.modules.warping_network import WarpingNetwork
                from live_portrait.modules.spade_generator import SPADEGenerator

                # 모델 로드
                appearance_path = self.model_dir / "appearance_feature_extractor.pth"
                motion_path = self.model_dir / "motion_extractor.pth"
                warping_path = self.model_dir / "warping_module.pth"
                spade_path = self.model_dir / "spade_generator.pth"

                if appearance_path.exists():
                    self._appearance_extractor = AppearanceFeatureExtractor()
                    self._appearance_extractor.load_state_dict(
                        torch.load(appearance_path, map_location=self.device)
                    )
                    self._appearance_extractor.to(self.device).eval()

                if motion_path.exists():
                    self._motion_extractor = MotionExtractor()
                    self._motion_extractor.load_state_dict(
                        torch.load(motion_path, map_location=self.device)
                    )
                    self._motion_extractor.to(self.device).eval()

                if warping_path.exists():
                    self._warping_module = WarpingNetwork()
                    self._warping_module.load_state_dict(
                        torch.load(warping_path, map_location=self.device)
                    )
                    self._warping_module.to(self.device).eval()

                if spade_path.exists():
                    self._spade_generator = SPADEGenerator()
                    self._spade_generator.load_state_dict(
                        torch.load(spade_path, map_location=self.device)
                    )
                    self._spade_generator.to(self.device).eval()

                # FP16 변환
                if self.fp16:
                    for module in [
                        self._appearance_extractor,
                        self._motion_extractor,
                        self._warping_module,
                        self._spade_generator,
                    ]:
                        if module is not None:
                            module.half()

                logger.info("LivePortrait models loaded successfully")

            except ImportError:
                logger.warning(
                    "LivePortrait package not installed. Using fallback implementation. "
                    "Install from: https://github.com/KwaiVGI/LivePortrait"
                )

            self._initialized = True
            logger.info("LivePortrait initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LivePortrait: {e}")
            return False

    async def extract_source_features(
        self,
        source_image: np.ndarray,
        source_id: str = "default",
    ) -> Dict[str, Any]:
        """
        소스 이미지에서 특징 추출

        Args:
            source_image: 소스 이미지 (BGR)
            source_id: 캐시 ID

        Returns:
            추출된 특징 딕셔너리
        """
        if not self._initialized:
            await self.initialize()

        # 캐시 확인
        if source_id in self._source_cache:
            return self._source_cache[source_id]

        features = {}

        try:
            import torch

            # 이미지 전처리
            img_tensor = self._preprocess_image(source_image)

            if self._appearance_extractor is not None:
                with torch.no_grad():
                    appearance_features = self._appearance_extractor(img_tensor)
                    features["appearance"] = appearance_features

            if self._motion_extractor is not None:
                with torch.no_grad():
                    source_motion = self._motion_extractor(img_tensor)
                    features["motion"] = source_motion

            features["source_image"] = source_image
            features["image_tensor"] = img_tensor

            # 캐시 저장
            self._source_cache[source_id] = features

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            features["source_image"] = source_image

        return features

    async def generate_frame(
        self,
        source_features: Dict[str, Any],
        motion_params: Dict[str, float],
    ) -> np.ndarray:
        """
        모션 파라미터를 적용하여 새 프레임 생성

        Args:
            source_features: extract_source_features의 결과
            motion_params: 모션 파라미터
                - head_pitch: 고개 상하 (-1 ~ 1)
                - head_yaw: 고개 좌우 (-1 ~ 1)
                - head_roll: 고개 회전 (-1 ~ 1)
                - expression_intensity: 표정 강도 (0 ~ 1)
                - blink: 눈 깜빡임 (0 ~ 1)
                - mouth_open: 입 벌림 (0 ~ 1)

        Returns:
            생성된 프레임
        """
        source_image = source_features.get("source_image")
        if source_image is None:
            raise ValueError("Source image not found in features")

        # 모델이 없으면 폴백 사용
        if self._warping_module is None:
            return self._apply_simple_animation(source_image, motion_params)

        try:
            import torch

            with torch.no_grad():
                # 모션 벡터 생성
                motion_vector = self._create_motion_vector(motion_params)

                # 소스 모션에 델타 적용
                source_motion = source_features.get("motion")
                if source_motion is not None:
                    target_motion = source_motion + motion_vector
                else:
                    target_motion = motion_vector

                # 워핑
                appearance = source_features.get("appearance")
                if appearance is not None and self._warping_module is not None:
                    warped = self._warping_module(
                        appearance,
                        source_motion,
                        target_motion,
                    )

                    # SPADE 생성
                    if self._spade_generator is not None:
                        output = self._spade_generator(warped)
                    else:
                        output = warped

                    # 후처리
                    return self._postprocess_output(output)

        except Exception as e:
            logger.error(f"Frame generation error: {e}")

        return self._apply_simple_animation(source_image, motion_params)

    async def generate_idle_sequence(
        self,
        source_image: np.ndarray,
        emotion: str = "neutral",
        duration_seconds: float = 5.0,
        fps: int = 30,
    ) -> List[np.ndarray]:
        """
        Idle 애니메이션 시퀀스 생성

        Args:
            source_image: 소스 이미지
            emotion: 감정 타입
            duration_seconds: 시퀀스 길이
            fps: 프레임 레이트

        Returns:
            프레임 시퀀스
        """
        if not self._initialized:
            await self.initialize()

        total_frames = int(duration_seconds * fps)
        frames = []

        # 소스 특징 추출
        features = await self.extract_source_features(source_image, f"idle_{emotion}")

        # 감정별 모션 프로필
        motion_profile = self._get_emotion_motion_profile(emotion)

        for frame_idx in range(total_frames):
            t = frame_idx / fps
            progress = frame_idx / total_frames

            # 시간에 따른 모션 파라미터 계산
            motion_params = self._calculate_idle_motion(
                t, progress, motion_profile
            )

            # 프레임 생성
            frame = await self.generate_frame(features, motion_params)
            frames.append(frame)

        return frames

    def _preprocess_image(self, image: np.ndarray) -> "torch.Tensor":
        """이미지 전처리"""
        import torch

        # 256x256으로 리사이즈
        resized = cv2.resize(image, (256, 256))

        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 정규화 및 텐서 변환
        tensor = torch.from_numpy(rgb).float() / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)

        if self.fp16:
            tensor = tensor.half()

        return tensor

    def _postprocess_output(self, output: "torch.Tensor") -> np.ndarray:
        """출력 후처리"""
        # 텐서 -> numpy
        output_np = output.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
        output_np = (output_np * 255).clip(0, 255).astype(np.uint8)

        # RGB -> BGR
        output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)

        return output_bgr

    def _create_motion_vector(
        self,
        params: Dict[str, float],
    ) -> "torch.Tensor":
        """모션 파라미터에서 벡터 생성"""
        import torch

        # 기본 모션 벡터 (21개 파라미터 예시)
        vector = torch.zeros(1, 21).to(self.device)

        # 파라미터 매핑
        vector[0, 0] = params.get("head_pitch", 0) * 0.3
        vector[0, 1] = params.get("head_yaw", 0) * 0.3
        vector[0, 2] = params.get("head_roll", 0) * 0.1
        vector[0, 10] = params.get("blink", 0) * 0.5
        vector[0, 15] = params.get("mouth_open", 0) * 0.4

        if self.fp16:
            vector = vector.half()

        return vector

    def _get_emotion_motion_profile(self, emotion: str) -> Dict[str, Any]:
        """감정별 모션 프로필"""
        profiles = {
            "neutral": {
                "head_movement": 0.02,
                "blink_frequency": 3.0,
                "expression_base": 0.0,
            },
            "happy": {
                "head_movement": 0.04,
                "blink_frequency": 4.0,
                "expression_base": 0.3,
                "mouth_curve": 0.2,
            },
            "sad": {
                "head_movement": 0.01,
                "blink_frequency": 2.0,
                "expression_base": -0.2,
                "head_tilt": -0.1,
            },
            "listening": {
                "head_movement": 0.03,
                "blink_frequency": 3.0,
                "nod_frequency": 1.5,
            },
            "thinking": {
                "head_movement": 0.02,
                "blink_frequency": 2.5,
                "look_up": 0.15,
            },
        }
        return profiles.get(emotion, profiles["neutral"])

    def _calculate_idle_motion(
        self,
        t: float,
        progress: float,
        profile: Dict[str, Any],
    ) -> Dict[str, float]:
        """시간에 따른 idle 모션 계산"""
        head_movement = profile.get("head_movement", 0.02)
        blink_freq = profile.get("blink_frequency", 3.0)

        params = {
            # 자연스러운 머리 움직임
            "head_pitch": np.sin(t * 0.5 * np.pi) * head_movement,
            "head_yaw": np.sin(t * 0.3 * np.pi) * head_movement,
            "head_roll": np.sin(t * 0.2 * np.pi) * head_movement * 0.5,

            # 표정 강도
            "expression_intensity": profile.get("expression_base", 0),

            # 눈 깜빡임 (주기적)
            "blink": self._calculate_blink(t, blink_freq),

            # 입 움직임 (미세)
            "mouth_open": abs(np.sin(t * 0.4 * np.pi)) * 0.02,
        }

        # 고개 끄덕임 (listening)
        if "nod_frequency" in profile:
            nod_freq = profile["nod_frequency"]
            params["head_pitch"] += np.sin(t * nod_freq * 2 * np.pi) * 0.05

        # 위 쳐다보기 (thinking)
        if "look_up" in profile:
            params["head_pitch"] -= profile["look_up"]

        return params

    def _calculate_blink(self, t: float, frequency: float) -> float:
        """눈 깜빡임 계산"""
        # 불규칙한 깜빡임 시뮬레이션
        blink_time = t * frequency
        blink_phase = blink_time % 1.0

        # 빠른 깜빡임 (0.1초)
        if blink_phase < 0.05:
            return blink_phase / 0.05
        elif blink_phase < 0.1:
            return 1.0 - (blink_phase - 0.05) / 0.05
        return 0.0

    def _apply_simple_animation(
        self,
        image: np.ndarray,
        params: Dict[str, float],
    ) -> np.ndarray:
        """
        간단한 애니메이션 (모델 없을 때 폴백)

        이미지 변환으로 움직임 시뮬레이션
        """
        h, w = image.shape[:2]
        result = image.copy()

        # 머리 움직임 시뮬레이션 (아핀 변환)
        pitch = params.get("head_pitch", 0) * 10
        yaw = params.get("head_yaw", 0) * 10

        # 변환 매트릭스
        M = np.float32([
            [1, 0, yaw],
            [0, 1, pitch],
        ])
        result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 눈 깜빡임 (상단 영역 어둡게)
        blink = params.get("blink", 0)
        if blink > 0.1:
            eye_region = result[int(h * 0.25):int(h * 0.4), :]
            darkness = int(blink * 50)
            eye_region = np.clip(eye_region.astype(np.int16) - darkness, 0, 255).astype(np.uint8)
            result[int(h * 0.25):int(h * 0.4), :] = eye_region

        return result

    def clear_cache(self) -> None:
        """소스 캐시 클리어"""
        self._source_cache.clear()

    async def cleanup(self) -> None:
        """리소스 정리"""
        self._appearance_extractor = None
        self._motion_extractor = None
        self._warping_module = None
        self._spade_generator = None
        self._source_cache.clear()
        self._initialized = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass

        logger.info("LivePortrait model cleaned up")
