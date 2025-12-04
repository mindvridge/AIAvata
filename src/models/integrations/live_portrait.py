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
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# LivePortrait 모델 경로
LIVE_PORTRAIT_MODEL_DIR = os.getenv("LIVE_PORTRAIT_MODEL_DIR", "models/live_portrait")
LIVE_PORTRAIT_SOURCE_DIR = os.getenv("LIVE_PORTRAIT_SOURCE_DIR", "external/LivePortrait")

# LivePortrait 루트 디렉토리를 Python 경로에 추가 (src 디렉토리가 아닌 루트)
_live_portrait_path = Path(LIVE_PORTRAIT_SOURCE_DIR).resolve()
if _live_portrait_path.exists() and str(_live_portrait_path) not in sys.path:
    sys.path.insert(0, str(_live_portrait_path))
    logger.info(f"Added LivePortrait root directory to Python path: {_live_portrait_path}")

    # __init__.py 파일이 없으면 생성 (패키지로 인식되도록)
    src_init = _live_portrait_path / "src" / "__init__.py"
    if not src_init.exists():
        try:
            src_init.touch()
            logger.info(f"Created {src_init}")
        except Exception:
            pass


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

        # LivePortrait 파이프라인
        self._pipeline = None
        self._cropper = None

        # 소스 이미지 캐시
        self._source_cache: Dict[str, Dict[str, Any]] = {}

        self._initialized = False
        self._use_fallback = False

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

            # LivePortrait 소스 경로 확인
            lp_base_path = Path(LIVE_PORTRAIT_SOURCE_DIR).resolve()
            lp_src_path = lp_base_path / "src"
            if not lp_src_path.exists():
                logger.warning(f"LivePortrait source not found: {lp_src_path}")
                self._use_fallback = True
                self._initialized = True
                return True

            try:
                # 모델 파일 경로 설정 (HuggingFace 다운로드 구조에 맞게)
                # 우선순위: liveportrait/base_models/*.pth > *.safetensors
                lp_base_models = self.model_dir / "liveportrait" / "base_models"
                lp_retarget = self.model_dir / "liveportrait" / "retargeting_models"

                # .pth 파일 경로 (HuggingFace 다운로드 구조)
                model_config_pth = {
                    "checkpoint_F": lp_base_models / "appearance_feature_extractor.pth",
                    "checkpoint_M": lp_base_models / "motion_extractor.pth",
                    "checkpoint_G": lp_base_models / "spade_generator.pth",
                    "checkpoint_W": lp_base_models / "warping_module.pth",
                    "checkpoint_S": lp_retarget / "stitching_retargeting_module.pth",
                }

                # .safetensors 파일 경로 (기존 구조)
                model_config_safetensors = {
                    "checkpoint_F": self.model_dir / "appearance_feature_extractor.safetensors",
                    "checkpoint_M": self.model_dir / "motion_extractor.safetensors",
                    "checkpoint_G": self.model_dir / "spade_generator.safetensors",
                    "checkpoint_W": self.model_dir / "warping_module.safetensors",
                    "checkpoint_S": self.model_dir / "retargeting_models" / "stitching_retargeting_module.safetensors",
                }

                # .pth 파일 먼저 확인
                if all(Path(p).exists() for p in model_config_pth.values()):
                    model_config = model_config_pth
                    logger.info("✅ Found LivePortrait models in liveportrait/base_models/ (.pth format)")
                elif all(Path(p).exists() for p in model_config_safetensors.values()):
                    model_config = model_config_safetensors
                    logger.info("✅ Found LivePortrait models in root directory (.safetensors format)")
                else:
                    # 어떤 파일들이 없는지 확인
                    missing_pth = [f"  - {k}: {v}" for k, v in model_config_pth.items() if not Path(v).exists()]
                    missing_safetensors = [f"  - {k}: {v}" for k, v in model_config_safetensors.items() if not Path(v).exists()]

                    logger.error("❌ LivePortrait 모델 파일이 없습니다!")
                    logger.error("   .pth 파일 (HuggingFace 다운로드 구조):")
                    for f in missing_pth:
                        logger.error(f)
                    logger.error("   .safetensors 파일 (기존 구조):")
                    for f in missing_safetensors:
                        logger.error(f)
                    logger.error("   다운로드 방법:")
                    logger.error("   python -c \"from huggingface_hub import snapshot_download; snapshot_download('KwaiVGI/LivePortrait', local_dir='models/live_portrait')\"")
                    logger.warning("LivePortrait model files not found. Using fallback.")
                    self._use_fallback = True
                    self._initialized = True
                    return True

                # 모델 파일이 있으면 계속 진행
                # LivePortrait를 패키지로 임포트 (sys.path에 루트 디렉토리 추가됨)
                try:
                    # LivePortrait 루트를 sys.path에 추가
                    lp_root = lp_src_path.parent
                    if str(lp_root) not in sys.path:
                        sys.path.insert(0, str(lp_root))

                    # __init__.py 파일 생성 (패키지로 인식되도록)
                    for subdir in ["", "config", "utils", "modules"]:
                        init_file = lp_src_path / subdir / "__init__.py" if subdir else lp_src_path / "__init__.py"
                        if not init_file.exists():
                            try:
                                init_file.touch()
                            except Exception:
                                pass

                    # 이제 패키지로 임포트
                    from src.config.inference_config import InferenceConfig
                    from src.config.crop_config import CropConfig
                    from src.live_portrait_pipeline import LivePortraitPipeline

                    logger.info("✅ LivePortrait 모듈 임포트 성공!")

                    # 설정 및 파이프라인 초기화
                    inference_cfg = InferenceConfig(
                        device_id=0 if self.device == "cuda" else -1,
                        flag_force_cpu=self.device != "cuda",
                    )
                    crop_cfg = CropConfig()

                    # 파이프라인 초기화
                    self._pipeline = LivePortraitPipeline(
                        inference_cfg=inference_cfg,
                        crop_cfg=crop_cfg
                    )
                    logger.info("✅ LivePortrait 파이프라인 초기화 성공!")

                except ImportError as e:
                    logger.error(f"❌ LivePortrait 모듈 임포트 실패: {e}")
                    logger.error("   해결방법: run.bat를 다시 실행하거나 LivePortrait 의존성을 설치하세요.")
                    logger.error("   pip install -r external/LivePortrait/requirements.txt")
                    self._use_fallback = True

                except Exception as e:
                    logger.error(f"❌ LivePortrait 파이프라인 초기화 실패: {e}")
                    logger.error("   Idle 애니메이션은 fallback 모드를 사용합니다.")
                    self._use_fallback = True

            except Exception as e:
                logger.error(f"❌ LivePortrait 설정 중 오류: {e}")
                logger.error("   해결방법: 모델 파일을 확인하세요 (models/live_portrait/)")
                self._use_fallback = True

            self._initialized = True
            if self._use_fallback:
                logger.warning("⚠️ LivePortrait가 fallback 모드로 실행됩니다. Idle 애니메이션이 제한됩니다.")
            else:
                logger.info("✅ LivePortrait initialization complete")
            return True

        except Exception as e:
            logger.error(f"❌ LivePortrait 초기화 실패: {e}")
            logger.error("   해결방법: 로그를 확인하고 모델 파일이 올바른지 확인하세요.")
            self._use_fallback = True
            self._initialized = True
            return True

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

        features = {"source_image": source_image}

        if not self._use_fallback and self._pipeline is not None:
            try:
                # LivePortrait 소스 특징 추출
                # RGB로 변환
                source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

                # 파이프라인으로 특징 추출
                source_info = self._pipeline.prepare_source(source_rgb)
                features["pipeline_source"] = source_info

                logger.debug(f"Extracted LivePortrait features for {source_id}")

            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}")

        # 캐시 저장
        self._source_cache[source_id] = features
        return features

    async def generate_idle_sequence(
        self,
        source_image: np.ndarray,
        emotion: str = "neutral",
        duration_seconds: float = 2.0,
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

        if not self._use_fallback and self._pipeline is not None and "pipeline_source" in features:
            # LivePortrait 파이프라인으로 프레임 생성
            try:
                for frame_idx in range(total_frames):
                    t = frame_idx / fps
                    motion_params = self._calculate_idle_motion(t, frame_idx / total_frames, motion_profile)

                    # 파이프라인으로 프레임 생성
                    frame = await self._generate_frame_with_pipeline(
                        features["pipeline_source"],
                        motion_params
                    )
                    frames.append(frame)

                return frames

            except Exception as e:
                logger.warning(f"Pipeline frame generation failed: {e}")
                # Fallback으로 전환

        # Fallback: 간단한 애니메이션
        for frame_idx in range(total_frames):
            t = frame_idx / fps
            motion_params = self._calculate_idle_motion(t, frame_idx / total_frames, motion_profile)
            frame = self._apply_simple_animation(source_image, motion_params)
            frames.append(frame)

        return frames

    async def _generate_frame_with_pipeline(
        self,
        source_info: Any,
        motion_params: Dict[str, float],
    ) -> np.ndarray:
        """LivePortrait 파이프라인으로 프레임 생성"""
        try:
            import torch

            # 모션 파라미터를 드라이빙 정보로 변환
            driving_info = self._create_driving_info(motion_params)

            # 파이프라인 추론
            result = self._pipeline.execute(source_info, driving_info)

            # 결과 이미지 추출 및 BGR 변환
            if result is not None:
                output_rgb = result["out"]
                output_bgr = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
                return output_bgr

        except Exception as e:
            logger.debug(f"Pipeline execution error: {e}")

        # 실패 시 원본 반환
        return source_info.get("source_image", np.zeros((512, 512, 3), dtype=np.uint8))

    def _create_driving_info(self, motion_params: Dict[str, float]) -> Dict[str, Any]:
        """모션 파라미터에서 드라이빙 정보 생성"""
        import torch

        # LivePortrait 드라이빙 형식
        driving_info = {
            "pitch": torch.tensor([[motion_params.get("head_pitch", 0) * 30]]),
            "yaw": torch.tensor([[motion_params.get("head_yaw", 0) * 30]]),
            "roll": torch.tensor([[motion_params.get("head_roll", 0) * 15]]),
            "exp": torch.zeros(1, 63),  # Expression coefficients
        }

        # 눈 깜빡임
        blink = motion_params.get("blink", 0)
        if blink > 0:
            driving_info["exp"][0, 0] = blink  # 눈 감김 coefficient

        # 입 열림
        mouth_open = motion_params.get("mouth_open", 0)
        if mouth_open > 0:
            driving_info["exp"][0, 25] = mouth_open  # 입 열림 coefficient

        return driving_info

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
            "surprised": {
                "head_movement": 0.03,
                "blink_frequency": 5.0,
                "expression_base": 0.4,
            },
            "angry": {
                "head_movement": 0.015,
                "blink_frequency": 2.0,
                "expression_base": -0.3,
            },
            "fearful": {
                "head_movement": 0.025,
                "blink_frequency": 6.0,
                "expression_base": 0.2,
            },
            "disgusted": {
                "head_movement": 0.01,
                "blink_frequency": 2.5,
                "expression_base": -0.25,
            },
            "sympathetic": {
                "head_movement": 0.02,
                "blink_frequency": 3.0,
                "head_tilt": 0.05,
                "nod_frequency": 0.5,
            },
            "concerned": {
                "head_movement": 0.015,
                "blink_frequency": 3.5,
                "expression_base": -0.1,
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
            # 자연스러운 머리 움직임 (사인/코사인 조합)
            "head_pitch": np.sin(t * 0.5 * np.pi) * head_movement + np.sin(t * 0.17 * np.pi) * head_movement * 0.5,
            "head_yaw": np.sin(t * 0.3 * np.pi) * head_movement + np.cos(t * 0.23 * np.pi) * head_movement * 0.3,
            "head_roll": np.sin(t * 0.2 * np.pi) * head_movement * 0.3,

            # 표정 강도
            "expression_intensity": profile.get("expression_base", 0),

            # 눈 깜빡임 (자연스러운 주기)
            "blink": self._calculate_blink(t, blink_freq),

            # 입 움직임 (호흡과 연동)
            "mouth_open": abs(np.sin(t * 0.4 * np.pi)) * 0.015,
        }

        # 고개 끄덕임 (listening)
        if "nod_frequency" in profile:
            nod_freq = profile["nod_frequency"]
            params["head_pitch"] += np.sin(t * nod_freq * 2 * np.pi) * 0.04

        # 위 쳐다보기 (thinking)
        if "look_up" in profile:
            params["head_pitch"] -= profile["look_up"] * 0.5

        # 고개 기울임
        if "head_tilt" in profile:
            params["head_roll"] += profile["head_tilt"]

        return params

    def _calculate_blink(self, t: float, frequency: float) -> float:
        """자연스러운 눈 깜빡임 계산"""
        # 불규칙한 깜빡임 시뮬레이션
        blink_time = t * frequency
        blink_phase = blink_time % 1.0

        # 빠른 깜빡임 (0.15초 주기)
        if blink_phase < 0.075:
            return blink_phase / 0.075
        elif blink_phase < 0.15:
            return 1.0 - (blink_phase - 0.075) / 0.075
        return 0.0

    def _apply_simple_animation(
        self,
        image: np.ndarray,
        params: Dict[str, float],
    ) -> np.ndarray:
        """
        간단한 애니메이션 (모델 없을 때 폴백)

        이미지 변환으로 움직임 시뮬레이션 - 눈에 보이도록 효과 증가
        """
        h, w = image.shape[:2]
        result = image.copy().astype(np.float32)

        # 머리 움직임 시뮬레이션 (아핀 변환) - 효과 증가
        pitch = params.get("head_pitch", 0) * 15  # 8 -> 15
        yaw = params.get("head_yaw", 0) * 15  # 8 -> 15
        roll = params.get("head_roll", 0) * 8  # 3 -> 8

        # 중심점
        center = (w // 2, h // 2)

        # 회전 + 이동 변환
        rotation_matrix = cv2.getRotationMatrix2D(center, roll, 1.0)
        rotation_matrix[0, 2] += yaw
        rotation_matrix[1, 2] += pitch

        result = cv2.warpAffine(
            result.astype(np.uint8),
            rotation_matrix,
            (w, h),
            borderMode=cv2.BORDER_REFLECT
        ).astype(np.float32)

        # 호흡 효과 (전체 밝기 미세 변화)
        breath = 1.0 + params.get("mouth_open", 0) * 0.05
        result = result * breath

        # 눈 깜빡임 효과 (상단 영역 어둡게) - 효과 증가
        blink = params.get("blink", 0)
        if blink > 0.1:
            eye_top = int(h * 0.25)
            eye_bottom = int(h * 0.4)
            eye_region = result[eye_top:eye_bottom, :].astype(np.float32)
            darkness = blink * 0.6  # 0.4 -> 0.6
            eye_region = eye_region * (1 - darkness)
            result[eye_top:eye_bottom, :] = eye_region

        return np.clip(result, 0, 255).astype(np.uint8)

    def clear_cache(self) -> None:
        """소스 캐시 클리어"""
        self._source_cache.clear()

    async def cleanup(self) -> None:
        """리소스 정리"""
        self._pipeline = None
        self._cropper = None
        self._source_cache.clear()
        self._initialized = False
        self._use_fallback = False

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, AttributeError):
            pass

        logger.info("LivePortrait model cleaned up")
