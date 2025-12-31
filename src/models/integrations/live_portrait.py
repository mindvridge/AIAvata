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
        driving_video_path: Optional[str] = None,
    ):
        """
        Initialize LivePortrait Model.

        Args:
            model_dir: 모델 파일 디렉토리
            device: 연산 디바이스
            fp16: FP16 추론 사용 여부
            driving_video_path: 드라이빙 비디오 경로 (idle 애니메이션용)
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.fp16 = fp16
        self.driving_video_path = driving_video_path

        # LivePortrait 파이프라인
        self._pipeline = None
        self._wrapper = None
        self._cropper = None

        # 소스 이미지 캐시
        self._source_cache: Dict[str, Dict[str, Any]] = {}

        # 드라이빙 비디오 모션 캐시
        self._driving_motions: Optional[List[Dict[str, Any]]] = None
        self._driving_video_fps: int = 30

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
                # importlib을 사용하여 LivePortrait 모듈 직접 로드 (AIAvata의 src와 충돌 방지)
                try:
                    import importlib.util

                    # __init__.py 파일 생성 (패키지로 인식되도록)
                    for subdir in ["", "config", "utils", "modules"]:
                        init_file = lp_src_path / subdir / "__init__.py" if subdir else lp_src_path / "__init__.py"
                        if not init_file.exists():
                            try:
                                init_file.touch()
                            except Exception:
                                pass

                    # 고유한 모듈 이름으로 LivePortrait 모듈 로드 (네임스페이스 충돌 방지)
                    def load_module_from_path(module_name: str, file_path: Path):
                        """파일 경로에서 모듈을 로드하는 헬퍼 함수"""
                        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
                        if spec is None or spec.loader is None:
                            raise ImportError(f"Cannot load module from {file_path}")
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        return module

                    # LivePortrait의 src를 lp_src라는 고유 이름으로 등록
                    lp_root = lp_src_path.parent
                    if str(lp_root) not in sys.path:
                        sys.path.insert(0, str(lp_root))

                    # 먼저 기본 유틸 모듈들을 로드 (의존성 순서대로)
                    # lp_src.config 패키지 설정
                    lp_config_init = lp_src_path / "config" / "__init__.py"
                    load_module_from_path("lp_src", lp_src_path / "__init__.py")
                    load_module_from_path("lp_src.config", lp_config_init)

                    # InferenceConfig 로드
                    inference_config_module = load_module_from_path(
                        "lp_src.config.inference_config",
                        lp_src_path / "config" / "inference_config.py"
                    )
                    InferenceConfig = inference_config_module.InferenceConfig

                    # CropConfig 로드
                    crop_config_module = load_module_from_path(
                        "lp_src.config.crop_config",
                        lp_src_path / "config" / "crop_config.py"
                    )
                    CropConfig = crop_config_module.CropConfig

                    logger.info("✅ LivePortrait config 모듈 임포트 성공!")

                    # 모델 파일 절대 경로 설정 (LivePortrait의 상대 경로 문제 해결)
                    model_dir_abs = self.model_dir.resolve()
                    lp_base_models_abs = model_dir_abs / "liveportrait" / "base_models"
                    lp_retarget_abs = model_dir_abs / "liveportrait" / "retargeting_models"
                    lp_config_abs = lp_src_path / "config"

                    # models.yaml 경로 (LivePortrait src/config 디렉토리에 있음)
                    models_yaml_path = lp_config_abs / "models.yaml"
                    if not models_yaml_path.exists():
                        logger.error(f"❌ models.yaml not found: {models_yaml_path}")
                        self._use_fallback = True
                        self._initialized = True
                        return True

                    # 설정 초기화 - 절대 경로로 모든 경로 지정
                    inference_cfg = InferenceConfig(
                        device_id=0 if self.device == "cuda" else -1,
                        flag_force_cpu=self.device != "cuda",
                        # models.yaml 절대 경로 지정
                        models_config=str(models_yaml_path),
                        # 체크포인트 절대 경로 지정
                        checkpoint_F=str(lp_base_models_abs / "appearance_feature_extractor.pth"),
                        checkpoint_M=str(lp_base_models_abs / "motion_extractor.pth"),
                        checkpoint_G=str(lp_base_models_abs / "spade_generator.pth"),
                        checkpoint_W=str(lp_base_models_abs / "warping_module.pth"),
                        checkpoint_S=str(lp_retarget_abs / "stitching_retargeting_module.pth"),
                    )
                    logger.info(f"   models.yaml 경로: {models_yaml_path}")
                    logger.info(f"   체크포인트 경로: {lp_base_models_abs}")

                    # Cropper 설정 - InsightFace 및 landmark 절대 경로 지정
                    insightface_root = model_dir_abs / "insightface"
                    landmark_path = model_dir_abs / "liveportrait" / "landmark.onnx"

                    # InsightFace 및 landmark 파일 존재 확인
                    insightface_models_dir = insightface_root / "models" / "buffalo_l"
                    required_insightface_files = [
                        insightface_models_dir / "det_10g.onnx",
                        insightface_models_dir / "2d106det.onnx",
                    ]
                    missing_files = []

                    if not landmark_path.exists():
                        missing_files.append(f"landmark.onnx: {landmark_path}")

                    for f in required_insightface_files:
                        if not f.exists():
                            missing_files.append(f"InsightFace: {f}")

                    if missing_files:
                        logger.warning("⚠️ 일부 모델 파일이 없습니다 (Cropper 비활성화):")
                        for f in missing_files:
                            logger.warning(f"   - {f}")
                        logger.info("   Cropper 없이 기본 파이프라인만 초기화합니다.")
                        # Cropper 없이 진행 (소스 이미지 크롭 불가)
                        crop_cfg = None
                    else:
                        crop_cfg = CropConfig(
                            insightface_root=str(insightface_root),
                            landmark_ckpt_path=str(landmark_path),
                            device_id=0 if self.device == "cuda" else -1,
                            flag_force_cpu=self.device != "cuda",
                        )
                        logger.info(f"   InsightFace 경로: {insightface_root}")
                        logger.info(f"   Landmark 경로: {landmark_path}")

                    # LivePortrait 모듈 로드 시도
                    try:
                        # utils 모듈들 먼저 등록
                        load_module_from_path("lp_src.utils", lp_src_path / "utils" / "__init__.py")
                        load_module_from_path("lp_src.modules", lp_src_path / "modules" / "__init__.py")

                        # LivePortraitWrapper 직접 로드 (더 간단한 방식)
                        wrapper_module = load_module_from_path(
                            "lp_src.live_portrait_wrapper",
                            lp_src_path / "live_portrait_wrapper.py"
                        )
                        LivePortraitWrapper = wrapper_module.LivePortraitWrapper

                        # Wrapper 초기화 (Cropper 없이도 동작)
                        self._wrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
                        logger.info("✅ LivePortraitWrapper 초기화 성공!")

                        # Cropper가 있으면 전체 파이프라인도 시도
                        if crop_cfg is not None:
                            try:
                                pipeline_module = load_module_from_path(
                                    "lp_src.live_portrait_pipeline",
                                    lp_src_path / "live_portrait_pipeline.py"
                                )
                                LivePortraitPipeline = pipeline_module.LivePortraitPipeline

                                self._pipeline = LivePortraitPipeline(
                                    inference_cfg=inference_cfg,
                                    crop_cfg=crop_cfg
                                )
                                logger.info("✅ LivePortrait 파이프라인 초기화 성공!")
                            except Exception as pipe_err:
                                logger.warning(f"⚠️ 파이프라인 초기화 실패, Wrapper만 사용: {pipe_err}")
                                self._pipeline = None
                        else:
                            self._pipeline = None
                            logger.info("   Cropper 없음 - Wrapper만 사용합니다.")

                    except Exception as pipeline_error:
                        logger.warning(f"⚠️ LivePortrait 모듈 로드 실패: {pipeline_error}")
                        import traceback
                        logger.warning(traceback.format_exc())
                        logger.warning("   Idle 애니메이션은 간단한 변환을 사용합니다.")
                        self._use_fallback = True

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

        # Wrapper나 Pipeline이 있으면 사용
        if not self._use_fallback and (self._wrapper is not None or self._pipeline is not None):
            try:
                # LivePortrait 소스 특징 추출
                # RGB로 변환
                source_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

                # Wrapper 직접 사용 (더 안정적)
                if self._wrapper is not None:
                    # 256x256으로 리사이즈
                    source_256 = cv2.resize(source_rgb, (256, 256))
                    I_s = self._wrapper.prepare_source(source_256)
                    x_s_info = self._wrapper.get_kp_info(I_s)
                    f_s = self._wrapper.extract_feature_3d(I_s)
                    x_s = self._wrapper.transform_keypoint(x_s_info)

                    features["wrapper_source"] = {
                        "I_s": I_s,
                        "x_s_info": x_s_info,
                        "f_s": f_s,
                        "x_s": x_s,
                        "source_256": source_256,
                    }
                    logger.debug(f"Extracted LivePortrait wrapper features for {source_id}")
                elif self._pipeline is not None:
                    # 파이프라인으로 특징 추출 (fallback)
                    source_info = self._pipeline.prepare_source(source_rgb)
                    features["pipeline_source"] = source_info
                    logger.debug(f"Extracted LivePortrait pipeline features for {source_id}")

            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())

        # 캐시 저장
        self._source_cache[source_id] = features
        return features

    async def load_driving_video(self, video_path: str) -> bool:
        """
        드라이빙 비디오 로드 및 모션 추출

        Args:
            video_path: 드라이빙 비디오 경로

        Returns:
            성공 여부
        """
        if not self._initialized:
            await self.initialize()

        if self._wrapper is None:
            logger.warning("LivePortrait wrapper not initialized, cannot load driving video")
            return False

        video_path = Path(video_path)
        if not video_path.exists():
            # 파일이 없어도 에러가 아닌 경고로 처리 (기본 동작에 영향을 주지 않음)
            logger.debug(f"Driving video not found (optional): {video_path}")
            return False

        logger.info(f"Loading driving video: {video_path}")

        try:
            import torch

            # 비디오 프레임 로드
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Cannot open driving video: {video_path}")
                return False

            self._driving_video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Driving video: {total_frames} frames at {self._driving_video_fps} fps")

            # 모션 추출
            self._driving_motions = []
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # BGR → RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 256x256으로 리사이즈
                frame_256 = cv2.resize(frame_rgb, (256, 256))

                # 모션 특징 추출
                try:
                    I_d = self._wrapper.prepare_source(frame_256)
                    x_d_info = self._wrapper.get_kp_info(I_d)

                    # 모션 정보 저장 (텐서를 CPU로 이동하여 저장)
                    motion_info = {}
                    for key, value in x_d_info.items():
                        if isinstance(value, torch.Tensor):
                            motion_info[key] = value.cpu().clone()
                        else:
                            motion_info[key] = value

                    self._driving_motions.append(motion_info)
                    frame_idx += 1

                    if frame_idx % 30 == 0:
                        logger.debug(f"Extracted motion from frame {frame_idx}/{total_frames}")

                except Exception as e:
                    logger.warning(f"Motion extraction failed for frame {frame_idx}: {e}")
                    continue

            cap.release()

            if len(self._driving_motions) > 0:
                logger.info(f"✅ Loaded {len(self._driving_motions)} motion frames from driving video")
                self.driving_video_path = str(video_path)
                return True
            else:
                logger.error("No motion frames extracted from driving video")
                return False

        except Exception as e:
            logger.error(f"Failed to load driving video: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def set_driving_video(self, video_path: str) -> None:
        """드라이빙 비디오 경로 설정 (초기화 시 로드됨)"""
        self.driving_video_path = video_path
        # 이미 초기화되었으면 바로 로드
        if self._initialized and self._wrapper is not None:
            asyncio.create_task(self.load_driving_video(video_path))

    def has_driving_video(self) -> bool:
        """드라이빙 비디오가 로드되어 있는지 확인"""
        return self._driving_motions is not None and len(self._driving_motions) > 0

    async def _generate_frame_with_driving(
        self,
        wrapper_source: Dict[str, Any],
        driving_frame_idx: int,
    ) -> np.ndarray:
        """
        드라이빙 비디오의 모션을 사용하여 프레임 생성 (상대적 모션 적용)

        LivePortrait 공식 파이프라인의 relative motion 방식:
        - 소스 이미지의 모션 + (드라이빙 프레임 모션 - 드라이빙 첫 프레임 모션)
        - 이렇게 하면 소스 이미지의 정체성을 유지하면서 드라이빙 비디오의 움직임만 전달

        Args:
            wrapper_source: 소스 이미지 특징
            driving_frame_idx: 드라이빙 프레임 인덱스

        Returns:
            생성된 프레임
        """
        if not self._driving_motions:
            raise ValueError("No driving motions loaded")

        try:
            import torch

            f_s = wrapper_source["f_s"]
            x_s = wrapper_source["x_s"]
            x_s_info = wrapper_source["x_s_info"]
            source_256 = wrapper_source["source_256"]

            # 드라이빙 모션 가져오기 (루프)
            motion_idx = driving_frame_idx % len(self._driving_motions)
            x_d_i_info_cpu = self._driving_motions[motion_idx]
            x_d_0_info_cpu = self._driving_motions[0]  # 첫 프레임 기준

            # 디바이스 확인
            device = x_s_info["pitch"].device if "pitch" in x_s_info else torch.device("cpu")
            dtype = x_s_info["pitch"].dtype if "pitch" in x_s_info else torch.float32

            # 드라이빙 모션을 GPU로 이동
            def to_device(info_cpu):
                info = {}
                for key, value in info_cpu.items():
                    if isinstance(value, torch.Tensor):
                        info[key] = value.to(device=device, dtype=dtype)
                    else:
                        info[key] = value
                return info

            x_d_i_info = to_device(x_d_i_info_cpu)
            x_d_0_info = to_device(x_d_0_info_cpu)

            # ===== 상대적 모션 계산 (핵심!) =====
            # 소스 모션 + (현재 드라이빙 프레임 - 첫 드라이빙 프레임)
            x_d_info_new = {}

            # 1. 회전 (R): R_new = R_d_i @ R_d_0.T @ R_s
            # pitch, yaw, roll을 사용하여 상대적 회전 계산
            for key in ['pitch', 'yaw', 'roll']:
                if key in x_s_info and key in x_d_i_info and key in x_d_0_info:
                    # 상대적 변화량 계산: source + (driving_i - driving_0)
                    delta = x_d_i_info[key] - x_d_0_info[key]
                    x_d_info_new[key] = x_s_info[key] + delta

            # 2. 표정 (exp): exp_new = exp_s + (exp_d_i - exp_d_0)
            if 'exp' in x_s_info and 'exp' in x_d_i_info and 'exp' in x_d_0_info:
                delta_exp = x_d_i_info['exp'] - x_d_0_info['exp']
                x_d_info_new['exp'] = x_s_info['exp'] + delta_exp

            # 3. 나머지 파라미터는 소스 값 사용
            for key in ['kp', 't', 'scale']:
                if key in x_s_info:
                    x_d_info_new[key] = x_s_info[key]

            # 키포인트 변환 (상대적 모션이 적용된 정보 사용)
            x_d = self._wrapper.transform_keypoint(x_d_info_new)

            # Stitching 적용 (얼굴 경계를 자연스럽게)
            if hasattr(self._wrapper, 'stitch') and hasattr(self._wrapper, 'stitching_retargeting_module'):
                try:
                    x_d_stitched = self._wrapper.stitch(x_s, x_d)
                    x_d = x_d + x_d_stitched
                except Exception as e:
                    logger.debug(f"Stitching skipped: {e}")

            # warp_decode로 프레임 생성
            ret_dct = self._wrapper.warp_decode(f_s, x_s, x_d)

            # 결과 파싱
            out = self._wrapper.parse_output(ret_dct['out'])[0]

            # BGR로 변환
            output_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            return output_bgr

        except Exception as e:
            logger.warning(f"Driving frame generation error: {e}")
            import traceback
            logger.warning(traceback.format_exc())

        # 실패 시 원본 반환
        source_256 = wrapper_source.get("source_256", np.zeros((256, 256, 3), dtype=np.uint8))
        return cv2.cvtColor(source_256, cv2.COLOR_RGB2BGR)

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

        # 🎬 드라이빙 비디오가 있으면 우선 사용 (최고 품질)
        if self.has_driving_video() and self._wrapper is not None and "wrapper_source" in features:
            try:
                logger.info(f"🎬 Using driving video for idle animation ({len(self._driving_motions)} motion frames)")
                for frame_idx in range(total_frames):
                    # 드라이빙 비디오 프레임으로 생성
                    frame = await self._generate_frame_with_driving(
                        features["wrapper_source"],
                        frame_idx
                    )
                    frames.append(frame)

                logger.info(f"✅ Generated {len(frames)} frames using driving video")
                return frames

            except Exception as e:
                logger.warning(f"Driving video frame generation failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # 수학적 모션으로 fallback

        # Wrapper로 프레임 생성 (수학적 모션)
        if not self._use_fallback and self._wrapper is not None and "wrapper_source" in features:
            try:
                for frame_idx in range(total_frames):
                    t = frame_idx / fps
                    motion_params = self._calculate_idle_motion(t, frame_idx / total_frames, motion_profile)

                    # Wrapper로 프레임 생성
                    frame = await self._generate_frame_with_wrapper(
                        features["wrapper_source"],
                        motion_params
                    )
                    frames.append(frame)

                return frames

            except Exception as e:
                logger.warning(f"Wrapper frame generation failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                # Fallback으로 전환

        # Pipeline으로 프레임 생성 (대안)
        elif not self._use_fallback and self._pipeline is not None and "pipeline_source" in features:
            try:
                for frame_idx in range(total_frames):
                    t = frame_idx / fps
                    motion_params = self._calculate_idle_motion(t, frame_idx / total_frames, motion_profile)

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

        # 실패 시 원본 반환 (source_image 크기 사용)
        source_image = source_info.get("source_image")
        if source_image is not None:
            return source_image
        else:
            # source_image가 없으면 기본값 (하지만 일반적으로는 없어서는 안 됨)
            logger.warning("source_image not found in source_info, using fallback")
            # 최소한의 기본 크기 (하지만 실제로는 source_image가 있어야 함)
            return np.zeros((1176, 784, 3), dtype=np.uint8)  # 784x1176 기본값

    async def _generate_frame_with_wrapper(
        self,
        wrapper_source: Dict[str, Any],
        motion_params: Dict[str, float],
    ) -> np.ndarray:
        """LivePortrait Wrapper로 프레임 생성 - 자연스러운 머리 회전과 표정 적용"""
        try:
            import torch
            import copy

            f_s = wrapper_source["f_s"]
            x_s = wrapper_source["x_s"]
            x_s_info = wrapper_source["x_s_info"]
            source_256 = wrapper_source["source_256"]

            # 모션 파라미터 추출
            head_pitch = motion_params.get("head_pitch", 0)
            head_yaw = motion_params.get("head_yaw", 0)
            head_roll = motion_params.get("head_roll", 0)
            blink = motion_params.get("blink", 0)
            mouth_open = motion_params.get("mouth_open", 0)

            # 디바이스 및 dtype 확인 (GPU 텐서와 CPU 텐서 혼합 방지)
            device = x_s_info["pitch"].device if "pitch" in x_s_info else torch.device("cpu")
            dtype = x_s_info["pitch"].dtype if "pitch" in x_s_info else torch.float32

            # x_s_info를 복사하여 수정 (원본 보존)
            # deepcopy는 GPU 텐서에서 문제가 있을 수 있으므로 수동 복사
            x_d_info = {}
            for key, value in x_s_info.items():
                if isinstance(value, torch.Tensor):
                    x_d_info[key] = value.clone()
                else:
                    x_d_info[key] = copy.deepcopy(value)

            # 1. 회전 적용 (pitch, yaw, roll) - 원본 값에 변화량 추가
            # LivePortrait의 회전 값은 라디안 단위
            # 중요: 텐서를 같은 device와 dtype으로 생성
            if "pitch" in x_d_info:
                x_d_info["pitch"] = x_s_info["pitch"] + torch.tensor([[head_pitch * 0.15]], device=device, dtype=dtype)
            if "yaw" in x_d_info:
                x_d_info["yaw"] = x_s_info["yaw"] + torch.tensor([[head_yaw * 0.15]], device=device, dtype=dtype)
            if "roll" in x_d_info:
                x_d_info["roll"] = x_s_info["roll"] + torch.tensor([[head_roll * 0.08]], device=device, dtype=dtype)

            # 2. 표정 적용 (눈 깜빡임, 입 움직임)
            if "exp" in x_d_info and x_d_info["exp"] is not None:
                exp = x_d_info["exp"].clone()
                exp_size = exp.shape[1]  # expression 계수 개수 확인

                # 눈 깜빡임 (exp의 처음 몇 개 계수)
                if blink > 0.1:
                    if exp_size > 0:
                        exp[0, 0] = blink * 0.5  # 왼쪽 눈
                    if exp_size > 1:
                        exp[0, 1] = blink * 0.5  # 오른쪽 눈

                # 입 열림 (LivePortrait exp 크기에 맞게 조정)
                # exp_size가 21이면 인덱스 14-20 영역 사용
                if mouth_open > 0:
                    mouth_idx = min(14, exp_size - 1)  # 입 관련 계수 (안전한 인덱스)
                    if mouth_idx >= 0:
                        exp[0, mouth_idx] = mouth_open * 0.3

                x_d_info["exp"] = exp

            # 새로운 키포인트 계산 (transform_keypoint 사용)
            x_d = self._wrapper.transform_keypoint(x_d_info)

            # warp_decode로 새 프레임 생성
            ret_dct = self._wrapper.warp_decode(f_s, x_s, x_d)

            # 결과 파싱
            out = self._wrapper.parse_output(ret_dct['out'])[0]  # HxWx3, uint8

            # BGR로 변환
            output_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            return output_bgr

        except Exception as e:
            logger.warning(f"Wrapper execution error: {e}")
            import traceback
            logger.warning(traceback.format_exc())

        # 실패 시 원본 반환
        source_256 = wrapper_source.get("source_256", np.zeros((256, 256, 3), dtype=np.uint8))
        return cv2.cvtColor(source_256, cv2.COLOR_RGB2BGR)

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
        self._wrapper = None
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
