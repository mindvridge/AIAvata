"""
MuseTalk 1.5 Integration Module.

실시간 립싱크 모델 통합
GitHub: https://github.com/TMElyralab/MuseTalk
라이선스: MIT (상업적 사용 가능)

특징:
- 실시간 오디오 기반 립싱크
- 30 FPS 출력
- GPU 가속 지원
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# MuseTalk 모델 경로
MUSETALK_MODEL_DIR = os.getenv("MUSETALK_MODEL_DIR", "models/musetalk")
MUSETALK_SOURCE_DIR = os.getenv("MUSETALK_SOURCE_DIR", "external/MuseTalk")

# MuseTalk 소스 디렉토리를 Python 경로에 추가
_musetalk_path = Path(MUSETALK_SOURCE_DIR).resolve()
if _musetalk_path.exists() and str(_musetalk_path) not in sys.path:
    sys.path.insert(0, str(_musetalk_path))
    logger.info(f"Added MuseTalk source directory to Python path: {_musetalk_path}")


class MuseTalkModel:
    """
    MuseTalk 1.5 립싱크 모델

    오디오를 입력받아 얼굴 이미지의 입 부분을 애니메이션합니다.
    """

    def __init__(
        self,
        model_dir: str = MUSETALK_MODEL_DIR,
        device: str = "cuda",
        fp16: bool = True,
        use_tensorrt: bool = False,
        tensorrt_workspace_size: int = 1024 * 1024 * 1024,  # 1GB
    ):
        """
        Initialize MuseTalk Model.

        Args:
            model_dir: 모델 파일 디렉토리
            device: 연산 디바이스
            fp16: FP16 추론 사용 여부
            use_tensorrt: TensorRT 사용 여부 (더 빠른 추론)
            tensorrt_workspace_size: TensorRT 워크스페이스 크기 (바이트)
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.fp16 = fp16
        self.use_tensorrt = use_tensorrt
        self.tensorrt_workspace_size = tensorrt_workspace_size

        # 모델 컴포넌트
        self._audio_processor = None
        self._unet = None
        self._unet_trt = None  # TensorRT 변환된 UNet
        self._vae = None
        self._face_parser = None
        self._positional_encoding = None  # PositionalEncoding for audio features
        self._whisper = None  # Whisper encoder 모델
        self._weight_dtype = None  # 모델 dtype (float16/float32)

        # 오디오 버퍼 (실시간 처리용)
        self._audio_buffer: List[np.ndarray] = []
        self._audio_buffer_size = 16000 * 2  # 2초 버퍼 (16kHz)

        # 캐시
        self._face_cache: Dict[str, Any] = {}

        self._initialized = False

    async def initialize(self) -> bool:
        """
        모델 초기화 및 로드

        Returns:
            초기화 성공 여부
        """
        if self._initialized:
            return True

        logger.info("Initializing MuseTalk model...")

        try:
            import torch
            from torch import nn

            # 모델 디렉토리 확인
            if not self.model_dir.exists():
                logger.warning(f"MuseTalk model directory not found: {self.model_dir}")
                logger.info("Creating placeholder for MuseTalk. Run 'python tools/setup_models.py' to download.")
                self.model_dir.mkdir(parents=True, exist_ok=True)
                self._initialized = True
                return True

            # MuseTalk 모델 로드 시도
            try:
                # 실제 MuseTalk 구현 임포트 시도
                from musetalk.utils.audio_processor import AudioProcessor
                from musetalk.models.unet import UNet
                from musetalk.models.vae import VAE
                from musetalk.utils.face_parsing import FaceParsing
                from musetalk.utils.utils import load_all_model

                logger.info("MuseTalk modules imported successfully")

                # 오디오 프로세서 초기화
                # Whisper 모델 자동 다운로드 (슬래시 없는 경로 사용)
                # AudioProcessor 기본값은 슬래시 포함 경로이므로 명시적으로 지정
                try:
                    self._audio_processor = AudioProcessor(feature_extractor_path="openai/whisper-tiny")
                    logger.info("AudioProcessor initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize AudioProcessor: {e}")
                    raise

                # 모델 디렉토리에서 실제 모델 파일 경로 찾기
                # 경로 확인: models/musetalk/musetalkV15
                model_version = "musetalkV15"
                
                # 여러 경로 시도
                possible_paths = [
                    self.model_dir / model_version,  # models/musetalk/musetalkV15
                    Path("models/musetalk") / model_version,  # 절대 경로
                    Path("models/musetalk/musetalkV15"),  # 직접 경로
                ]
                
                model_base_dir = None
                for path in possible_paths:
                    if path.exists():
                        model_base_dir = path
                        logger.info(f"Found MuseTalk model directory: {model_base_dir}")
                        break
                
                if model_base_dir and model_base_dir.exists():
                    # musetalk.json 설정 파일 찾기
                    config_path = model_base_dir / "musetalk.json"
                    unet_path = model_base_dir / "unet.pth"

                    # 필수 파일 확인
                    missing_files = []
                    if not config_path.exists():
                        missing_files.append(f"  - musetalk.json: {config_path}")
                    if not unet_path.exists():
                        missing_files.append(f"  - unet.pth: {unet_path}")

                    if missing_files:
                        logger.error("❌ MuseTalk 필수 모델 파일이 없습니다!")
                        logger.error("   누락된 파일:")
                        for f in missing_files:
                            logger.error(f)
                        logger.error("   다운로드 방법: run.bat 다시 실행 또는:")
                        logger.error("   python -c \"from huggingface_hub import snapshot_download; snapshot_download('TMElyralab/MuseTalk', local_dir='models/musetalk/hf_download')\"")
                        self._initialized = True
                        return False

                    if config_path.exists() and unet_path.exists():
                        logger.info(f"Loading MuseTalk models from {model_base_dir}")
                        
                        # UNet 모델 로드
                        device_obj = torch.device(self.device if torch.cuda.is_available() else "cpu")
                        
                        # UNet 초기화 (실제 MuseTalk 구조)
                        self._unet = UNet(
                            unet_config=str(config_path),
                            model_path=str(unet_path),
                            use_float16=self.fp16,
                            device=device_obj
                        )
                        logger.info("UNet model loaded successfully")
                        
                        # TensorRT 변환 시도 (선택적)
                        if self.use_tensorrt:
                            try:
                                from ...utils.tensorrt_utils import (
                                    convert_unet_to_tensorrt,
                                    check_tensorrt_available,
                                )
                                
                                if check_tensorrt_available():
                                    logger.info("🚀 Attempting TensorRT conversion for UNet...")
                                    
                                    # 엔진 저장 디렉토리
                                    engine_dir = self.model_dir / "tensorrt_engines"
                                    
                                    # 샘플 입력 생성 (실제 추론 시 사용되는 형태)
                                    sample_latent = torch.randn(
                                        1, 8, 32, 32,
                                        dtype=torch.float16 if self.fp16 else torch.float32,
                                        device=device_obj
                                    )
                                    sample_timesteps = torch.tensor([0], dtype=torch.long, device=device_obj)
                                    sample_encoder_hidden_states = torch.randn(
                                        1, 50, 384,
                                        dtype=torch.float16 if self.fp16 else torch.float32,
                                        device=device_obj
                                    )
                                    
                                    # TensorRT로 변환
                                    self._unet_trt = convert_unet_to_tensorrt(
                                        unet_model=self._unet.model,
                                        sample_latent=sample_latent,
                                        sample_timesteps=sample_timesteps,
                                        sample_encoder_hidden_states=sample_encoder_hidden_states,
                                        engine_dir=engine_dir,
                                        fp16=self.fp16,
                                        workspace_size=self.tensorrt_workspace_size,
                                    )
                                    
                                    if self._unet_trt is not None:
                                        logger.info("✅ UNet TensorRT conversion successful! Using TensorRT for inference.")
                                    else:
                                        logger.warning("⚠️ TensorRT conversion failed, falling back to PyTorch inference")
                                else:
                                    # TensorRT가 사용 불가능하지만 use_tensorrt=True로 설정된 경우에만 경고
                                    # 기본값은 False이므로 경고 대신 INFO로 변경
                                    logger.info("ℹ️ TensorRT not available, using PyTorch inference (normal if TensorRT is not installed)")
                            except Exception as e:
                                logger.warning(f"⚠️ TensorRT conversion failed: {e}")
                                logger.warning("Falling back to PyTorch inference")
                                import traceback
                                logger.debug(traceback.format_exc())
                        
                        # PositionalEncoding 초기화 (오디오 특징용)
                        from musetalk.models.unet import PositionalEncoding
                        self._positional_encoding = PositionalEncoding(d_model=384)
                        device_obj = torch.device(self.device if torch.cuda.is_available() else "cpu")
                        if self.fp16:
                            self._positional_encoding = self._positional_encoding.half()
                        self._positional_encoding.to(device_obj)
                        logger.info("PositionalEncoding initialized")
                        
                        # Weight dtype 설정
                        self._weight_dtype = torch.float16 if self.fp16 else torch.float32
                        
                        # Whisper 모델 초기화 (실시간 오디오 특징 추출용)
                        try:
                            from transformers import WhisperModel
                            whisper_model_path = "openai/whisper-tiny"
                            
                            # 여러 경로 시도
                            whisper_paths = [
                                Path("models/whisper"),
                                Path("external/MuseTalk/models/whisper"),
                                "openai/whisper-tiny",  # HuggingFace에서 자동 다운로드
                            ]
                            
                            whisper_path = None
                            for wp in whisper_paths:
                                if isinstance(wp, str):
                                    # HuggingFace 모델 ID
                                    whisper_path = wp
                                    break
                                elif wp.exists():
                                    whisper_path = str(wp)
                                    break
                            
                            if whisper_path:
                                logger.info(f"Loading Whisper model from: {whisper_path}")
                                self._whisper = WhisperModel.from_pretrained(whisper_path)
                                self._whisper = self._whisper.to(device=device_obj, dtype=self._weight_dtype).eval()
                                self._whisper.requires_grad_(False)
                                logger.info("Whisper model loaded successfully")
                            else:
                                logger.warning("Whisper model path not found, using feature_extractor only")
                                self._whisper = None
                        except Exception as e:
                            logger.warning(f"Failed to load Whisper model: {e}. Using feature_extractor only.")
                            self._whisper = None
                        
                        # VAE 모델 로드 (SD-VAE)
                        # 여러 경로 시도
                        vae_paths = [
                            Path("models/sd-vae-ft-mse"),  # 다운로드된 경로
                            model_base_dir.parent / "sd-vae-ft-mse",
                            Path("models/vae"),
                            model_base_dir.parent / "vae",
                        ]
                        
                        vae_path = None
                        for vp in vae_paths:
                            if vp.exists():
                                vae_path = vp
                                logger.info(f"Found VAE model directory: {vae_path}")
                                break
                        
                        if vae_path and vae_path.exists():
                            try:
                                logger.info(f"Loading VAE model from {vae_path}...")
                                self._vae = VAE(
                                    model_path=str(vae_path),
                                    resized_img=256,
                                    use_float16=self.fp16
                                )
                                logger.info(f"✅ VAE model loaded successfully from {vae_path}")
                            except Exception as e:
                                logger.warning(f"VAE model load failed: {e}")
                                logger.warning("Continuing without VAE (quality may be reduced)")
                                self._vae = None
                        else:
                            logger.warning(f"VAE model directory not found in: {[str(p) for p in vae_paths]}")
                            logger.warning("Continuing without VAE (quality may be reduced)")
                            self._vae = None
                        
                        # Face Parsing 초기화 (선택적, device 인자 없음)
                        # FaceParsing은 './models/face-parse-bisent/79999_iter.pth' 경로 기대
                        try:
                            import shutil
                            face_parser_model_path = Path("./models/face-parse-bisent/79999_iter.pth")
                            face_parser_model_path.parent.mkdir(parents=True, exist_ok=True)

                            # 모델 파일이 없으면 여러 경로에서 복사 시도
                            if not face_parser_model_path.exists():
                                source_paths = [
                                    Path(self.model_dir) / "face-parse-bisent" / "79999_iter.pth",
                                    Path("models/musetalk/face-parse-bisent/79999_iter.pth"),
                                    Path("models/musetalk/hf_download/models/face-parse-bisent/79999_iter.pth"),
                                ]
                                for src_path in source_paths:
                                    if src_path.exists():
                                        logger.info(f"Copying face parser model from {src_path}")
                                        shutil.copy(str(src_path), str(face_parser_model_path))
                                        break

                            # resnet18 모델도 확인
                            resnet_path = Path("./models/face-parse-bisent/resnet18-5c106cde.pth")
                            if not resnet_path.exists():
                                resnet_sources = [
                                    Path(self.model_dir) / "face-parse-bisent" / "resnet18-5c106cde.pth",
                                    Path("models/musetalk/face-parse-bisent/resnet18-5c106cde.pth"),
                                ]
                                for src_path in resnet_sources:
                                    if src_path.exists():
                                        shutil.copy(str(src_path), str(resnet_path))
                                        break

                            if face_parser_model_path.exists():
                                self._face_parser = FaceParsing()
                                logger.info("=" * 50)
                                logger.info("✅ Face Parser 초기화 성공")
                                logger.info("   모델: models/face-parse-bisent/79999_iter.pth")
                                logger.info("=" * 50)
                            else:
                                logger.error("=" * 60)
                                logger.error("❌ FACE PARSER 모델 없음 - 립싱크 품질이 저하됩니다!")
                                logger.error("   누락 파일: models/face-parse-bisent/79999_iter.pth")
                                logger.error("")
                                logger.error("   다운로드 방법:")
                                logger.error("   1. https://huggingface.co/vivym/face-parsing-bisenet 접속")
                                logger.error("   2. 79999_iter.pth 다운로드")
                                logger.error("   3. models/face-parse-bisent/ 폴더에 저장")
                                logger.error("=" * 60)
                                self._face_parser = None
                        except Exception as e:
                            logger.error("=" * 60)
                            logger.error(f"❌ FACE PARSER 초기화 실패: {e}")
                            logger.error("   립싱크 품질이 저하됩니다!")
                            logger.error("=" * 60)
                            self._face_parser = None

                        logger.info("MuseTalk models loaded successfully")
                    else:
                        logger.warning(f"MuseTalk model files not found: config={config_path.exists()}, unet={unet_path.exists()}")
                        raise FileNotFoundError(f"MuseTalk model files not found")
                else:
                    logger.warning(f"MuseTalk model directory not found: {model_base_dir}")
                    raise FileNotFoundError(f"MuseTalk model directory not found")

            except ImportError:
                logger.warning(
                    "MuseTalk package not installed. Using fallback implementation. "
                    "Install from: https://github.com/TMElyralab/MuseTalk"
                )
                # 폴백: 기본 오디오 처리기 사용
                self._audio_processor = FallbackAudioProcessor()

            self._initialized = True
            logger.info("MuseTalk initialization complete")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize MuseTalk: {e}")
            return False

    async def process_frame(
        self,
        source_frame: np.ndarray,
        audio_chunk: np.ndarray,
        audio_sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        오디오에 맞춰 프레임에 립싱크 적용

        Args:
            source_frame: 원본 프레임 (BGR, HxWxC)
            audio_chunk: 오디오 청크 (float32)
            audio_sample_rate: 오디오 샘플레이트

        Returns:
            립싱크 적용된 프레임
        """
        if not self._initialized:
            await self.initialize()

        # 🔑 입력 프레임 크기 로깅 (첫 프레임만)
        h, w = source_frame.shape[:2]
        logger.debug(f"📐 MuseTalk process_frame 입력: {w}x{h}")

        # 모델이 로드되지 않았으면 에러 표시
        if self._unet is None:
            logger.error("❌ MuseTalk UNet 모델이 로드되지 않았습니다!")
            logger.error("   필요한 파일: models/musetalk/musetalkV15/unet.pth")
            logger.error("   해결방법: run.bat를 다시 실행하거나 수동으로 다운로드하세요.")
            return source_frame

        try:
            import torch

            # 오디오 특징 추출
            audio_energy = np.sqrt(np.mean(audio_chunk**2)) if len(audio_chunk) > 0 else 0.0
            logger.debug(f"🎤 Extracting audio features: shape={audio_chunk.shape}, sample_rate={audio_sample_rate}, energy={audio_energy:.4f}")
            audio_features = self._extract_audio_features(audio_chunk, audio_sample_rate)
            if hasattr(audio_features, 'mean') and hasattr(audio_features, 'std'):
                logger.debug(f"🎤 Audio features extracted: shape={audio_features.shape}, mean={audio_features.mean().item():.4f}, std={audio_features.std().item():.4f}")
            else:
                logger.debug(f"🎤 Audio features extracted: shape={audio_features.shape}")

            # VAE가 없으면 에러 표시
            if self._vae is None:
                logger.error("❌ MuseTalk VAE 모델이 로드되지 않았습니다!")
                logger.error("   필요한 파일: models/musetalk/sd-vae-ft-mse/")
                logger.error("   해결방법: run.bat를 다시 실행하거나 수동으로 다운로드하세요.")
                return source_frame
            
            # 얼굴 영역 추출 (MediaPipe로 얼굴 감지 후 크롭)
            import cv2

            # MediaPipe로 얼굴 위치 찾기 (캐시 사용)
            # 아바타 이미지는 항상 같으므로 첫 번째 성공 결과를 캐시
            cache_key = f"face_bbox_{source_frame.shape}"
            if cache_key in self._face_cache:
                face_bbox = self._face_cache[cache_key]
                logger.debug(f"Using cached face_bbox: {face_bbox}")
            else:
                face_bbox = self._detect_face_bbox(source_frame)
                if face_bbox is not None:
                    self._face_cache[cache_key] = face_bbox
                    logger.info(f"✅ Face detected and cached: {face_bbox}")
                else:
                    # 🔑 폴백: 아바타 이미지의 기본 얼굴 위치 (중앙)
                    h, w = source_frame.shape[:2]
                    # 얼굴이 이미지 중앙에 있다고 가정 (패딩 포함)
                    face_size = int(min(w, h) * 0.7)  # 이미지의 70%
                    cx, cy = w // 2, h // 2
                    x1 = max(0, cx - face_size // 2)
                    y1 = max(0, cy - face_size // 2)
                    x2 = min(w, x1 + face_size)
                    y2 = min(h, y1 + face_size)
                    face_bbox = (x1, y1, x2, y2)
                    self._face_cache[cache_key] = face_bbox
                    logger.warning(f"⚠️ 얼굴 감지 실패 - 기본 위치 사용: {face_bbox}")

            # 패딩 정보 저장 (블렌딩 시 사용)
            pad_info = None  # (pad_x, pad_y, resized_w, resized_h, face_h, face_w)
            
            if face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                # 얼굴 영역만 크롭
                face_region = source_frame[y1:y2, x1:x2]
                face_h, face_w = face_region.shape[:2]
                face_aspect = face_w / face_h
                
                # 🔑 비율 유지하면서 256x256으로 변환 (패딩 추가)
                # 비율 유지하면서 256 크기에 맞춤
                if face_aspect > 1.0:  # 가로가 더 긴 경우
                    new_w = 256
                    new_h = int(256 / face_aspect)
                else:  # 세로가 더 긴 경우
                    new_h = 256
                    new_w = int(256 * face_aspect)
                
                # 비율 유지 리사이즈
                face_resized = cv2.resize(face_region, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                # 256x256에 맞추기 위해 패딩 추가
                face_crop = np.zeros((256, 256, 3), dtype=np.uint8)
                pad_y = (256 - new_h) // 2
                pad_x = (256 - new_w) // 2
                face_crop[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = face_resized
                
                # 패딩 정보 저장
                pad_info = (pad_x, pad_y, new_w, new_h, face_h, face_w)
                
                logger.debug(f"📐 얼굴 크롭 비율 유지: 원본={face_w}x{face_h} (비율={face_aspect:.4f}), 리사이즈={new_w}x{new_h}, 패딩=({pad_x},{pad_y})")
                logger.debug(f"Face detected: bbox=({x1}, {y1}, {x2}, {y2}), aspect={face_aspect:.2f}, padded to 256x256 (pad={pad_x},{pad_y})")
            else:
                # 얼굴 감지 실패 시 전체 프레임 사용 (비율 유지)
                h, w = source_frame.shape[:2]
                aspect = w / h
                
                if aspect > 1.0:
                    new_w = 256
                    new_h = int(256 / aspect)
                else:
                    new_h = 256
                    new_w = int(256 * aspect)
                
                frame_resized = cv2.resize(source_frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                
                face_crop = np.zeros((256, 256, 3), dtype=np.uint8)
                pad_y = (256 - new_h) // 2
                pad_x = (256 - new_w) // 2
                face_crop[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = frame_resized
                
                # 패딩 정보 저장 (전체 프레임용)
                pad_info = (pad_x, pad_y, new_w, new_h, h, w)
                face_bbox = None
                logger.debug(f"📐 전체 프레임 비율 유지: 원본={w}x{h} (비율={aspect:.4f}), 리사이즈={new_w}x{new_h}, 패딩=({pad_x},{pad_y})")
                logger.debug(f"Face not detected, using full frame (aspect={aspect:.2f}, padded to 256x256, pad={pad_x},{pad_y})")
            
            # VAE로 얼굴 이미지를 latent로 인코딩
            # get_latents_for_unet은 이미지 경로나 numpy array를 받을 수 있음
            device_obj = torch.device(self.device if torch.cuda.is_available() else "cpu")
            latent_input = self._vae.get_latents_for_unet(face_crop)
            latent_input = latent_input.to(device_obj)
            if self._weight_dtype:
                latent_input = latent_input.to(dtype=self._weight_dtype)
            else:
                if self.fp16:
                    latent_input = latent_input.half()

            # 오디오 특징 처리 (MuseTalk 방식)
            # _extract_audio_features가 이미 [batch, 50, 384] 형태를 반환
            device_obj = torch.device(self.device if torch.cuda.is_available() else "cpu")
            timesteps = torch.tensor([0], device=device_obj)
            
            logger.debug(f"Audio features from extractor: {audio_features.shape}")
            
            # PositionalEncoding 적용 (MuseTalk 방식)
            # 입력: [batch, 50, 384], 출력: [batch, 50, 384]
            if self._positional_encoding and len(audio_features.shape) == 3:
                audio_features = self._positional_encoding(audio_features.to(device_obj))
                logger.debug(f"Audio features after PE: {audio_features.shape}")
            
            # UNet 추론 (실제 MuseTalk 방식)
            with torch.no_grad():
                # latent_input 준비
                if latent_input.dtype != self._unet.model.dtype:
                    latent_input = latent_input.to(dtype=self._unet.model.dtype)
                
                # audio_features를 device로 이동
                audio_features = audio_features.to(device_obj)
                if self._weight_dtype:
                    audio_features = audio_features.to(dtype=self._weight_dtype)
                
                # UNet의 encoder_hidden_states는 [batch, seq_len, hidden_dim] 형태
                # MuseTalk에서는 get_whisper_chunk 결과가 [batch, (c h) w] 형태인데
                # 이것이 PositionalEncoding을 거쳐서 [batch, seq_len, features]가 됨
                # 실시간에서는 [batch, 1, features] 형태를 사용
                
                logger.debug(f"UNet input - latent_input: {latent_input.shape}, audio_features: {audio_features.shape}")
                
                try:
                    # UNet 추론 (MuseTalk realtime_inference.py 방식)
                    # TensorRT가 있으면 사용, 없으면 PyTorch 모델 사용
                    logger.debug(f"🔄 UNet inference starting - latent: {latent_input.shape}, audio: {audio_features.shape}")
                    
                    if self._unet_trt is not None:
                        # TensorRT 추론
                        logger.debug("Using TensorRT for UNet inference")
                        # torch-tensorrt로 변환된 모델은 일반 PyTorch 모델처럼 사용 가능
                        unet_output = self._unet_trt(
                            latent_input,
                            timesteps,
                            audio_features  # encoder_hidden_states 대신 positional argument
                        )
                        # TensorRT 출력은 직접 tensor이거나 BaseOutput일 수 있음
                        if hasattr(unet_output, 'sample'):
                            pred_latents = unet_output.sample
                        else:
                            pred_latents = unet_output
                    else:
                        # PyTorch 추론
                        logger.debug("Using PyTorch for UNet inference")
                        unet_output = self._unet.model(
                            latent_input,
                            timesteps,
                            encoder_hidden_states=audio_features
                        )
                        # .sample 속성 접근 (UNet2DConditionModel의 반환값)
                        pred_latents = unet_output.sample

                    logger.debug(f"✅ UNet output: {pred_latents.shape}")
                except Exception as e:
                    logger.error(f"❌ UNet 추론 실패: {e}")
                    logger.error(f"   latent_input shape: {latent_input.shape}, dtype: {latent_input.dtype}")
                    logger.error(f"   audio_features shape: {audio_features.shape}, dtype: {audio_features.dtype}")
                    logger.error(f"   timesteps: {timesteps}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()}")
                    logger.error("   해결방법: UNet 모델 파일이 올바른지 확인하세요 (models/musetalk/musetalkV15/unet.pth)")
                    return source_frame
                
                # VAE 디코딩
                vae_device = next(self._vae.vae.parameters()).device
                vae_dtype = next(self._vae.vae.parameters()).dtype
                logger.debug(f"🔄 VAE decoding starting... (VAE device={vae_device}, dtype={vae_dtype})")
                pred_latents = pred_latents.to(dtype=self._vae.vae.dtype)
                recon = self._vae.decode_latents(pred_latents)
                logger.debug(f"✅ VAE decoded: type={type(recon)}, shape={recon.shape if hasattr(recon, 'shape') else 'N/A'}")
                
                # 첫 번째 프레임만 사용 (배치 크기 1)
                if isinstance(recon, (list, tuple)):
                    output = recon[0]
                elif len(recon.shape) == 4 and recon.shape[0] > 1:
                    output = recon[0:1]  # 첫 번째만
                else:
                    output = recon

            # 후처리: VAE 출력은 BGR 이미지이므로 그대로 사용
            # output은 numpy array [H, W, 3] 형태
            output_np = None
            try:
                if isinstance(output, torch.Tensor):
                    output_np = output.detach().cpu().numpy()
                    if len(output_np.shape) == 4:  # [batch, H, W, C]
                        output_np = output_np[0]  # 첫 번째만
                    elif len(output_np.shape) == 3 and output_np.shape[0] == 3:  # [C, H, W]
                        output_np = output_np.transpose(1, 2, 0)  # [H, W, C]
                    
                    # 0-1 범위를 0-255로 변환
                    if output_np.max() <= 1.0:
                        output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
                    else:
                        output_np = output_np.clip(0, 255).astype(np.uint8)
                elif isinstance(output, np.ndarray):
                    output_np = output.copy()
                    # 4D 배열인 경우 배치 차원 제거
                    if len(output_np.shape) == 4:  # [batch, H, W, C]
                        output_np = output_np[0]  # 첫 번째만
                    elif len(output_np.shape) == 3 and output_np.shape[0] == 3:  # [C, H, W]
                        output_np = output_np.transpose(1, 2, 0)  # [H, W, C]
                    # 0-1 범위를 0-255로 변환
                    if output_np.max() <= 1.0:
                        output_np = (output_np * 255).clip(0, 255).astype(np.uint8)
                    elif output_np.dtype != np.uint8:
                        output_np = output_np.clip(0, 255).astype(np.uint8)
                else:
                    logger.warning(f"Unexpected output type: {type(output)}, returning source frame")
                    return source_frame
                    
                # 출력 형태 검증
                if output_np is None or output_np.size == 0:
                    logger.warning("VAE output is None or empty after processing, returning source frame")
                    return source_frame
                    
            except Exception as e:
                logger.error(f"VAE output post-processing failed: {e}", exc_info=True)
                return source_frame
            
            # 원본 크기로 리사이즈
            h, w = source_frame.shape[:2]
            
            # output_np 유효성 검사
            if output_np is None or output_np.size == 0:
                logger.warning("VAE output is None or empty, returning source frame")
                return source_frame
            
            # output_np 형태 검사 및 정규화
            if len(output_np.shape) < 2:
                logger.warning(f"Invalid output_np shape (too few dimensions): {output_np.shape}, returning source frame")
                return source_frame
            
            # 높이와 너비 확인
            output_h, output_w = output_np.shape[:2]
            if output_h <= 0 or output_w <= 0:
                logger.warning(f"Invalid output_np dimensions: h={output_h}, w={output_w}, returning source frame")
                return source_frame
            
            # 리사이즈 시도
            try:
                # VAE 출력은 이미 256x256이어야 함 (확인 후 필요시에만 리사이즈)
                if output_np.shape[:2] != (256, 256):
                    logger.warning(f"VAE 출력 크기가 예상과 다름: {output_np.shape[:2]}, 256x256으로 리사이즈")
                    result_256 = cv2.resize(output_np, (256, 256), interpolation=cv2.INTER_CUBIC)
                else:
                    result_256 = output_np.copy()
                
                source_256 = face_crop.copy()
                
                # 패딩 정보 로깅
                if pad_info is not None:
                    pad_x, pad_y, resized_w, resized_h, orig_h, orig_w = pad_info
                    logger.debug(f"📐 패딩 정보: pad=({pad_x},{pad_y}), resized=({resized_w}x{resized_h}), orig=({orig_w}x{orig_h})")
                else:
                    logger.warning("⚠️ pad_info가 None입니다 - 비율 왜곡 가능성")

                # VAE 출력과 원본의 차이 로깅
                diff = result_256.astype(np.float32) - source_256.astype(np.float32)

                # 🔑 입 영역 좌표 수정 (256x256 기준)
                # 입 중앙으로 조정 (155-210 → 145-195)
                mouth_y1, mouth_y2 = 145, 195  # Y: 145-195 (중심 170)
                mouth_x1, mouth_x2 = 70, 186   # X: 70-186 (중심 128)

                mouth_region_diff = np.abs(diff[mouth_y1:mouth_y2, mouth_x1:mouth_x2])  # 입 영역만
                avg_diff_mouth = np.mean(mouth_region_diff)
                max_diff_mouth = np.max(mouth_region_diff)

                # 🔑 차이가 작으면 입 영역만 부드럽게 증폭
                if avg_diff_mouth < 25.0:
                    # 증폭 계수 증가 (2.5~4.0배) - 입 모양 크게
                    amplify_factor = max(2.5, min(4.0, 50.0 / max(avg_diff_mouth, 1.0)))

                    # 입 영역만 선택적 증폭 (마스크 생성)
                    mouth_mask = np.zeros((256, 256), dtype=np.float32)

                    # 🔑 부드러운 그라디언트 마스크 (입 중앙 집중)
                    for y in range(mouth_y1, mouth_y2):
                        for x in range(mouth_x1, mouth_x2):
                            # 중앙에서 멀어질수록 약해짐
                            dy = (y - (mouth_y1 + mouth_y2) // 2) / ((mouth_y2 - mouth_y1) / 2)
                            dx = (x - (mouth_x1 + mouth_x2) // 2) / ((mouth_x2 - mouth_x1) / 2)
                            dist = np.sqrt(dx**2 + dy**2)
                            # 0.8로 더 부드러운 페이드
                            mouth_mask[y, x] = max(0, 1.0 - dist * 0.8)
                    
                    # 입 영역만 증폭 적용
                    diff_selective = diff.copy()
                    for c in range(3):
                        diff_selective[:, :, c] = diff[:, :, c] * (1.0 + (amplify_factor - 1.0) * mouth_mask)
                    
                    result_256 = np.clip(source_256.astype(np.float32) + diff_selective, 0, 255).astype(np.uint8)
                    logger.debug(f"🔊 입 영역 차이 증폭: {avg_diff_mouth:.1f} → x{amplify_factor:.2f} (선택적)")
                else:
                    logger.debug(f"✅ VAE 출력 차이: 입 영역={avg_diff_mouth:.1f}, max={max_diff_mouth:.1f}")

                # =====================================================
                # MuseTalk 원래 블렌딩 방식 사용
                # =====================================================
                from PIL import Image
                
                # 🔑 MuseTalk blending.py의 get_image_blending 방식 적용
                # 원본 프레임과 VAE 출력을 PIL Image로 변환
                body_pil = Image.fromarray(source_frame[:, :, ::-1])  # BGR to RGB
                face_pil = Image.fromarray(result_256[:, :, ::-1])    # BGR to RGB
                
                if face_bbox is not None:
                    x1, y1, x2, y2 = face_bbox
                    logger.debug(f"🔍 Face bbox: ({x1}, {y1}, {x2}, {y2})")
                    
                    # MuseTalk 방식: Face Parser로 마스크 생성 (mode="jaw")
                    mask_array = None
                    
                    # Face Parser 상태 추적
                    face_parser_error_reason = None
                    
                    if self._face_parser is not None:
                        logger.debug("🔍 Face Parser 시작...")
                        try:
                            # 확장된 얼굴 영역 crop (expand=1.2)
                            expand = 1.2
                            x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2
                            face_w, face_h = x2 - x1, y2 - y1
                            s = int(max(face_w, face_h) // 2 * expand)
                            
                            # crop box 계산
                            crop_x1 = max(0, x_c - s)
                            crop_y1 = max(0, y_c - s)
                            crop_x2 = min(source_frame.shape[1], x_c + s)
                            crop_y2 = min(source_frame.shape[0], y_c + s)
                            crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
                            
                            # 확장된 얼굴 영역 crop
                            face_large = body_pil.crop(crop_box)
                            ori_shape = face_large.size
                            
                            logger.debug(f"Face Parser 입력: size={ori_shape}, mode=jaw")

                            # Face Parser로 마스크 생성 (mode="jaw" - 공식 MuseTalk 설정)
                            seg_image = self._face_parser(face_large, mode="jaw")
                            
                            if seg_image is not None:
                                logger.debug(f"🔍 Face Parser 출력: type={type(seg_image).__name__}, size={seg_image.size if hasattr(seg_image, 'size') else 'N/A'}")
                                seg_image = seg_image.resize(ori_shape)
                                
                                # face_box 영역만 추출
                                mask_small = seg_image.crop((
                                    x1 - crop_x1, y1 - crop_y1,
                                    x2 - crop_x1, y2 - crop_y1
                                ))
                                
                                # 전체 마스크 생성
                                mask_image = Image.new('L', ori_shape, 0)
                                mask_image.paste(mask_small, (x1 - crop_x1, y1 - crop_y1))
                                
                                # 🔑 상단 50% 제거 (공식 MuseTalk upper_boundary_ratio=0.5)
                                width, height = mask_image.size
                                top_boundary = int(height * 0.5)
                                modified_mask = Image.new('L', ori_shape, 0)
                                modified_mask.paste(
                                    mask_image.crop((0, top_boundary, width, height)),
                                    (0, top_boundary)
                                )

                                # 🔑 블러 강도 적절히 조정 (0.15 → 0.10)
                                # 색상 매칭이 추가되어 블러는 적절히만
                                blur_kernel_size = int(0.10 * ori_shape[0] // 2 * 2) + 1
                                if blur_kernel_size < 9:
                                    blur_kernel_size = 9
                                mask_array = cv2.GaussianBlur(
                                    np.array(modified_mask),
                                    (blur_kernel_size, blur_kernel_size), 0
                                )
                                
                                # 🔑 마스크 최대값 255 (완전 불투명)
                                # 중앙 영역은 VAE 결과를 그대로 사용
                                mask_array = np.clip(mask_array, 0, 255)
                                
                                # 마스크 유효 픽셀 수 확인
                                valid_pixels = np.sum(mask_array > 0)
                                if valid_pixels < 100:
                                    face_parser_error_reason = f"마스크 픽셀 부족 ({valid_pixels}개)"
                                    mask_array = None
                                else:
                                    logger.debug(f"✅ Face Parser 성공: blur={blur_kernel_size}, pixels={valid_pixels}, max={np.max(mask_array):.0f}")
                            else:
                                face_parser_error_reason = "Face Parser가 None 반환"
                                logger.error("❌ Face Parser가 None 반환!")
                                mask_array = None
                        except Exception as e:
                            face_parser_error_reason = f"예외 발생: {str(e)}"
                            import traceback
                            logger.error(f"❌ Face Parser 예외: {e}")
                            logger.error(traceback.format_exc())
                            mask_array = None
                    else:
                        face_parser_error_reason = "Face Parser 모델이 로드되지 않음"
                        logger.error("❌ Face Parser 모델이 로드되지 않음!")
                    
                    # Face Parser 실패 시 MuseTalk 스타일 폴백 마스크
                    if mask_array is None:
                        logger.error("=" * 60)
                        logger.error("❌ FACE PARSER 실패 - 폴백 마스크 사용")
                        logger.error(f"   원인: {face_parser_error_reason}")
                        logger.error("   해결방법:")
                        logger.error("   1. models/face-parse-bisent/79999_iter.pth 확인")
                        logger.error("   2. models/face-parse-bisent/resnet18-5c106cde.pth 확인")
                        logger.error("=" * 60)
                        
                        expand = 1.2
                        x_c, y_c = (x1 + x2) // 2, (y1 + y2) // 2
                        face_w, face_h = x2 - x1, y2 - y1
                        s = int(max(face_w, face_h) // 2 * expand)
                        
                        crop_x1 = max(0, x_c - s)
                        crop_y1 = max(0, y_c - s)
                        crop_x2 = min(source_frame.shape[1], x_c + s)
                        crop_y2 = min(source_frame.shape[0], y_c + s)
                        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
                        
                        face_large = body_pil.crop(crop_box)
                        ori_shape = face_large.size
                        
                        # 🔑 수정: 입 영역만 작은 마스크 (전체 하단 X)
                        mask_h, mask_w = ori_shape[1], ori_shape[0]
                        mask_array = np.zeros((mask_h, mask_w), dtype=np.uint8)
                        
                        # 얼굴 영역의 상대 위치 계산
                        rel_x1 = x1 - crop_x1
                        rel_y1 = y1 - crop_y1
                        rel_x2 = x2 - crop_x1
                        rel_y2 = y2 - crop_y1
                        
                        # 🔑 입 영역만 정확하게 마스크 (더 좁은 범위)
                        face_height = rel_y2 - rel_y1
                        face_width = rel_x2 - rel_x1
                        
                        # 입 위치: 얼굴 하단 65-80% 영역 (입술만 정확하게)
                        mouth_y_start = rel_y1 + int(face_height * 0.65)
                        mouth_y_end = rel_y1 + int(face_height * 0.80)
                        
                        # 입 폭: 얼굴 폭의 30% (더 좁게)
                        mouth_x_center = (rel_x1 + rel_x2) // 2
                        mouth_half_width = int(face_width * 0.15)
                        mouth_x_start = max(0, mouth_x_center - mouth_half_width)
                        mouth_x_end = min(mask_w, mouth_x_center + mouth_half_width)
                        
                        # 타원형 마스크 생성 (더 자연스러움)
                        mouth_center_y = (mouth_y_start + mouth_y_end) // 2
                        mouth_radius_y = (mouth_y_end - mouth_y_start) // 2
                        mouth_radius_x = mouth_half_width
                        
                        for y in range(mouth_y_start, mouth_y_end):
                            for x in range(mouth_x_start, mouth_x_end):
                                # 타원 거리 계산
                                dy = (y - mouth_center_y) / max(1, mouth_radius_y)
                                dx = (x - mouth_x_center) / max(1, mouth_radius_x)
                                dist = np.sqrt(dx**2 + dy**2)
                                
                                if dist <= 1.0:
                                    # 중앙에서 멀어질수록 페이드
                                    intensity = int(200 * (1.0 - dist * 0.5))  # 최대 200
                                    mask_array[y, x] = max(mask_array[y, x], intensity)
                        
                        # 🔑 블러 강도 적절히 조정 (0.15 → 0.10)
                        blur_kernel_size = int(0.10 * ori_shape[0] // 2 * 2) + 1
                        if blur_kernel_size < 9:
                            blur_kernel_size = 9
                        mask_array = cv2.GaussianBlur(mask_array, (blur_kernel_size, blur_kernel_size), 0)

                        # 🔑 마스크 최대값 255 (완전 불투명)
                        mask_array = np.clip(mask_array, 0, 255)
                    
                    # =====================================================
                    # MuseTalk get_image_blending 방식으로 블렌딩
                    # =====================================================
                    # 🔑 패딩 제거 후 원본 비율로 리사이즈 (비율 왜곡 방지)
                    if pad_info is not None:
                        pad_x, pad_y, resized_w, resized_h, orig_face_h, orig_face_w = pad_info
                        logger.debug(f"📐 패딩 제거: result_256 shape={result_256.shape}, pad=({pad_x},{pad_y}), extract=({resized_w}x{resized_h})")
                        
                        # 패딩 제거: 원본 비율로 리사이즈된 영역만 추출
                        result_without_pad = result_256[pad_y:pad_y+resized_h, pad_x:pad_x+resized_w]
                        
                        # 원본 얼굴 크기로 리사이즈 (비율 유지)
                        target_size = (orig_face_w, orig_face_h)
                        result_face = cv2.resize(result_without_pad, target_size, interpolation=cv2.INTER_CUBIC)
                        
                        # 비율 검증
                        restored_aspect = result_face.shape[1] / result_face.shape[0]
                        expected_aspect = orig_face_w / orig_face_h
                        aspect_diff = abs(restored_aspect - expected_aspect)
                        
                        logger.debug(f"📐 복원 결과: {result_face.shape[1]}x{result_face.shape[0]}, 비율={restored_aspect:.4f}, 예상={expected_aspect:.4f}, 차이={aspect_diff:.4f}")
                    else:
                        # 패딩 정보가 없으면 기존 방식 (하지만 일반적으로는 pad_info가 있어야 함)
                        logger.warning(f"⚠️ pad_info 없음 - 강제 리사이즈: {result_256.shape} → ({x2-x1}x{y2-y1})")
                        result_face = cv2.resize(result_256, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

                    # 🔑 Unsharp Mask 비활성화 (아티팩트 방지)
                    # 샤프닝이 입 주변 노이즈를 증폭시켜 아티팩트 유발
                    # gaussian = cv2.GaussianBlur(result_face, (0, 0), 2.0)
                    # result_face = cv2.addWeighted(result_face, 1.0, gaussian, 0.0, 0)
                    # result_face = np.clip(result_face, 0, 255).astype(np.uint8)

                    # 🔑 색상 매칭 - VAE 출력 색상을 원본에 맞춤 (흰색 라인 방지)
                    # 원본 얼굴 영역 추출
                    original_face_region = source_frame[y1:y2, x1:x2]
                    if original_face_region.shape[:2] == result_face.shape[:2]:
                        # 각 채널별 평균 밝기 계산
                        for c in range(3):
                            orig_mean = np.mean(original_face_region[:, :, c])
                            result_mean = np.mean(result_face[:, :, c])
                            if result_mean > 0:
                                # 밝기 차이 보정 (80% 적용으로 자연스럽게)
                                scale = 1.0 + (orig_mean - result_mean) / (result_mean + 1e-6) * 0.8
                                result_face[:, :, c] = np.clip(result_face[:, :, c] * scale, 0, 255).astype(np.uint8)
                        logger.debug(f"🎨 색상 매칭 적용: orig_mean vs result_mean 보정")

                    # 🔑 크기 검증 및 강제 맞춤 (찌그러짐 방지)
                    expected_w, expected_h = x2 - x1, y2 - y1
                    actual_h, actual_w = result_face.shape[:2]

                    if actual_w != expected_w or actual_h != expected_h:
                        logger.warning(f"⚠️ 크기 불일치 감지: 실제=({actual_w}x{actual_h}), 예상=({expected_w}x{expected_h})")
                        result_face = cv2.resize(result_face, (expected_w, expected_h), interpolation=cv2.INTER_CUBIC)
                        logger.info(f"✅ 크기 강제 조정: ({expected_w}x{expected_h})")

                    result_face_pil = Image.fromarray(result_face[:, :, ::-1])
                    
                    # face_large에 result_face 붙이기
                    face_large.paste(result_face_pil, (x1 - crop_x1, y1 - crop_y1))
                    
                    # 🔑 마스크 조작 제거 - 원본 마스크 그대로 사용 (선명도 향상)
                    # np.where 조작이 입 영역에 불필요한 아티팩트를 생성해서 제거
                    mask_pil = Image.fromarray(mask_array).convert("L")

                    # 마스크 통계 로깅
                    mask_min = np.min(mask_array)
                    mask_max = np.max(mask_array)
                    mask_mean = np.mean(mask_array)
                    mask_nonzero = np.count_nonzero(mask_array)
                    logger.debug(f"🎭 블렌딩 마스크: min={mask_min}, max={mask_max}, mean={mask_mean:.1f}, nonzero={mask_nonzero}")
                    
                    body_pil.paste(face_large, (crop_x1, crop_y1), mask_pil)
                    
                    # numpy로 변환
                    result_frame = np.array(body_pil)[:, :, ::-1]  # RGB to BGR
                    
                    # 크기 검증 (source_frame과 동일해야 함)
                    result_h, result_w = result_frame.shape[:2]
                    source_h, source_w = source_frame.shape[:2]
                    if result_h != source_h or result_w != source_w:
                        logger.warning(f"⚠️ MuseTalk 출력 크기 불일치: 결과={result_w}x{result_h}, 원본={source_w}x{source_h}, 리사이즈 적용")
                        result_frame = cv2.resize(result_frame, (source_w, source_h), interpolation=cv2.INTER_CUBIC)
                    
                    logger.debug(f"✅ MuseTalk lip sync SUCCESS: output shape={result_frame.shape}")
                    return result_frame

                # 얼굴 bbox가 없는 경우 (폴백) - 전체 프레임에 VAE 출력 적용
                else:
                    logger.error("❌ face_bbox가 None - 얼굴 감지 실패! 전체 프레임에 VAE 적용")
                    # 🔑 패딩 제거 후 원본 비율로 리사이즈 (비율 왜곡 방지)
                    if pad_info is not None:
                        pad_x, pad_y, resized_w, resized_h, orig_h, orig_w = pad_info
                        # 패딩 제거: 원본 비율로 리사이즈된 영역만 추출
                        result_without_pad = result_256[pad_y:pad_y+resized_h, pad_x:pad_x+resized_w]
                        # 원본 프레임 크기로 리사이즈 (비율 유지)
                        result_frame = cv2.resize(result_without_pad, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
                    else:
                        # 패딩 정보가 없으면 기존 방식
                        result_frame = cv2.resize(result_256, (w, h), interpolation=cv2.INTER_CUBIC)

                    if result_frame is not None and result_frame.shape[:2] == (h, w):
                        logger.debug(f"✅ MuseTalk lip sync SUCCESS: output shape={result_frame.shape}")
                        return result_frame
                    else:
                        logger.warning(f"Resize result invalid: {result_frame.shape if result_frame is not None else None}, returning source frame")
                        return source_frame
            except Exception as e:
                logger.error(f"OpenCV resize/blend failed: {e}, output_np shape: {output_np.shape}, target size: ({w}, {h}), returning source frame")
                return source_frame

        except Exception as e:
            logger.error(f"❌ MuseTalk 추론 오류: {e}", exc_info=True)
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            logger.error("   해결방법: 로그를 확인하고 모델 파일이 올바른지 확인하세요.")
            return source_frame

    def _extract_audio_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> "torch.Tensor":
        """
        오디오에서 특징 추출 (MuseTalk 방식: Whisper encoder 사용)
        
        MuseTalk 원본 형태: [batch, 50, 384]
        - 50 = 10 time steps × 5 whisper layers
        - 384 = hidden dimension
        
        Args:
            audio: 오디오 배열 (float32, [-1, 1])
            sample_rate: 오디오 샘플레이트
            
        Returns:
            Whisper encoder hidden states (torch.Tensor) [batch, 50, 384]
        """
        import torch
        from einops import rearrange

        device_obj = torch.device(self.device if torch.cuda.is_available() else "cpu")
        TARGET_SEQ_LEN = 50  # MuseTalk 기대 형태: [batch, 50, 384]
        HIDDEN_DIM = 384

        if self._audio_processor is None:
            logger.warning("⚠️ AudioProcessor가 없어서 폴백 특징 사용")
            return self._generate_fallback_features(device_obj, TARGET_SEQ_LEN, HIDDEN_DIM)

        try:
            import librosa
            
            # 16000 Hz로 리샘플링 (MuseTalk 요구사항)
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # 최소 길이 보장 (Whisper 요구사항: 최소 0.5초)
            min_length = int(16000 * 0.5)
            if len(audio) < min_length:
                audio = np.pad(audio, (0, min_length - len(audio)), mode='constant', constant_values=0)
            
            # Whisper feature extractor로 mel spectrogram 추출
            audio_feature = self._audio_processor.feature_extractor(
                audio,
                return_tensors="pt",
                sampling_rate=16000
            ).input_features
            
            audio_feature = audio_feature.to(device_obj)
            if self._weight_dtype:
                audio_feature = audio_feature.to(dtype=self._weight_dtype)
            
            # Whisper encoder로 hidden states 추출
            if self._whisper is not None:
                with torch.no_grad():
                    encoder_output = self._whisper.encoder(
                        audio_feature, 
                        output_hidden_states=True
                    )
                    
                    hidden_states = encoder_output.hidden_states
                    # hidden_states는 tuple: (layer0, layer1, ..., layerN)
                    # 각 레이어: [batch, seq_len, hidden_dim]
                    
                    num_layers = len(hidden_states)
                    
                    # 첫 번째 레이어 형태 확인
                    first_layer = hidden_states[0]
                    batch_size, seq_len, hidden_dim = first_layer.shape
                    
                    logger.debug(f"🎤 Whisper: {num_layers} layers, seq_len={seq_len}, hidden_dim={hidden_dim}")
                    
                    # MuseTalk은 5개 레이어 × 10 time steps = 50 사용
                    # Whisper-tiny는 5개 레이어 (embedding + 4 encoder layers)
                    target_layers = 5
                    target_time_steps = 10
                    
                    # 레이어 선택 (마지막 5개 또는 가능한 만큼)
                    if num_layers >= target_layers:
                        selected_layers = hidden_states[-target_layers:]
                    else:
                        # 레이어 수가 부족하면 반복 패딩
                        selected_layers = list(hidden_states)
                        while len(selected_layers) < target_layers:
                            selected_layers.append(hidden_states[-1])  # 마지막 레이어 반복
                        selected_layers = selected_layers[:target_layers]
                    
                    num_selected = len(selected_layers)
                    logger.debug(f"Selected {num_selected} layers for audio features")
                    
                    # 🔑 오디오 에너지 기반 위치 선택 (보수적으로)
                    # 오디오 에너지가 높을수록 다른 위치 선택 (lip movement variation)
                    audio_energy = float(np.sqrt(np.mean(audio ** 2)))
                    
                    # 에너지를 0~1로 정규화 (0.001~0.3 범위 가정)
                    energy_normalized = min(1.0, max(0.0, (audio_energy - 0.001) / 0.3))
                    
                    # 에너지에 따라 시작 위치 결정 (시간적 다양성 추가)
                    if seq_len >= target_time_steps:
                        # 중앙 기준으로 변동
                        center = seq_len // 2
                        # 에너지와 작은 랜덤 오프셋으로 위치 다양화
                        import random
                        random_offset = random.randint(-20, 20)  # 작은 랜덤 변화
                        energy_offset = int(energy_normalized * (seq_len - target_time_steps) * 0.3)
                        position_offset = energy_offset + random_offset
                        start_idx = max(0, min(seq_len - target_time_steps, center - target_time_steps // 2 + position_offset))
                        end_idx = start_idx + target_time_steps
                        
                        layer_features = []
                        for layer in selected_layers:
                            layer_slice = layer[:, start_idx:end_idx, :]  # [batch, 10, hidden_dim]
                            # 🔑 오디오 에너지 기반 스케일링 (1.0 ~ 1.8) - 입 움직임 강조
                            scale_factor = 1.0 + energy_normalized * 0.8
                            layer_slice = layer_slice * scale_factor
                            layer_features.append(layer_slice)
                        
                        logger.debug(f"Audio energy={audio_energy:.4f}, norm={energy_normalized:.2f}, scale={scale_factor:.2f}, pos={start_idx}")
                    else:
                        # seq_len이 부족하면 반복 패딩으로 확장
                        layer_features = []
                        scale_factor = 1.0 + energy_normalized * 0.8  # 에너지 기반 스케일링
                        for layer in selected_layers:
                            repeat_times = (target_time_steps + seq_len - 1) // seq_len
                            layer_repeated = layer.repeat(1, repeat_times, 1)[:, :target_time_steps, :]
                            layer_repeated = layer_repeated * scale_factor
                            layer_features.append(layer_repeated)
                    
                    # 레이어들을 결합: [batch, 10, 384] × 5 -> [batch, 50, 384]
                    stacked = torch.stack(layer_features, dim=2)  # [batch, 10, 5, hidden_dim]
                    audio_feats_final = rearrange(stacked, 'b t l h -> b (t l) h')  # [batch, 50, hidden_dim]
                    
                    # hidden_dim이 384가 아니면 조정
                    if hidden_dim != HIDDEN_DIM:
                        logger.warning(f"Hidden dim mismatch: {hidden_dim} vs {HIDDEN_DIM}, padding/truncating")
                        if hidden_dim < HIDDEN_DIM:
                            # 패딩
                            padding = torch.zeros(batch_size, audio_feats_final.shape[1], HIDDEN_DIM - hidden_dim, 
                                                device=device_obj, dtype=audio_feats_final.dtype)
                            audio_feats_final = torch.cat([audio_feats_final, padding], dim=2)
                        else:
                            # 잘라내기
                            audio_feats_final = audio_feats_final[:, :, :HIDDEN_DIM]
                    
                    # seq_len 조정 (50 보장)
                    if audio_feats_final.shape[1] < TARGET_SEQ_LEN:
                        padding_size = TARGET_SEQ_LEN - audio_feats_final.shape[1]
                        padding = audio_feats_final[:, :padding_size, :].clone()
                        audio_feats_final = torch.cat([audio_feats_final, padding], dim=1)
                    elif audio_feats_final.shape[1] > TARGET_SEQ_LEN:
                        audio_feats_final = audio_feats_final[:, :TARGET_SEQ_LEN, :]
                    
                    logger.debug(f"✅ Audio features: {audio_feats_final.shape}")  # [1, 50, 384]
                    return audio_feats_final
            else:
                logger.error("❌ Whisper encoder가 로드되지 않음!")
                logger.error("   해결 방법: 서버 재시작 또는 Whisper 모델 확인")
                return self._generate_fallback_features(device_obj, TARGET_SEQ_LEN, HIDDEN_DIM)
                
        except Exception as e:
            logger.error(f"Audio feature extraction failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return self._generate_fallback_features(device_obj, TARGET_SEQ_LEN, HIDDEN_DIM)
    
    def _generate_fallback_features(self, device, seq_len: int = 50, hidden_dim: int = 384) -> "torch.Tensor":
        """폴백 오디오 특징 생성 (립싱크가 작동하지 않음 - 중립 포즈)"""
        import torch
        # 중립 포즈를 위해 0으로 채움 (랜덤 노이즈 대신)
        # 이렇게 하면 최소한 입이 이상하게 움직이지는 않음
        logger.warning("⚠️ 폴백 특징 사용 중 - 립싱크 비활성화 (중립 포즈)")
        fallback = torch.zeros(1, seq_len, hidden_dim, device=device)
        if self._weight_dtype:
            fallback = fallback.to(dtype=self._weight_dtype)
        return fallback

    def _simple_audio_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """간단한 오디오 특징 추출 (폴백)"""
        try:
            import librosa

            # MFCC 추출
            mfcc = librosa.feature.mfcc(
                y=audio.astype(np.float32),
                sr=sample_rate,
                n_mfcc=13,
            )
            return mfcc.T  # (time, features)

        except ImportError:
            # librosa 없으면 에너지 기반 특징
            frame_size = int(sample_rate * 0.025)  # 25ms
            hop_size = int(sample_rate * 0.010)  # 10ms

            features = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                energy = np.sqrt(np.mean(frame ** 2))
                features.append([energy] * 13)

            return np.array(features, dtype=np.float32)

    def _detect_face_bbox(
        self,
        frame: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        MediaPipe로 얼굴 영역 감지 (tasks API 우선, solutions API 폴백)

        Args:
            frame: 입력 프레임 (BGR)

        Returns:
            (x1, y1, x2, y2) 얼굴 bounding box 또는 None
        """
        import cv2
        import os
        import urllib.request

        h, w = frame.shape[:2]

        try:
            import mediapipe as mp
            
            # ========================================
            # 1. tasks API 사용 (MediaPipe 0.10.0+)
            # ========================================
            try:
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core import base_options

                # 모델 파일 경로 - 프로젝트 루트 기준 절대 경로
                import os
                project_root = Path(__file__).parent.parent.parent.parent  # src/models/integrations -> project root
                model_path = project_root / "models" / "face_detection_short_range.tflite"
                model_path.parent.mkdir(parents=True, exist_ok=True)

                if not model_path.exists():
                    logger.info("📥 Face detection 모델 다운로드 중...")
                    url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
                    urllib.request.urlretrieve(url, str(model_path))
                    logger.info(f"✅ 모델 다운로드 완료: {model_path}")

                # 🔑 절대 경로 문자열로 변환 (mediapipe 경로 문제 방지)
                model_path_str = str(model_path.absolute())
                base_opts = base_options.BaseOptions(model_asset_path=model_path_str)
                options = vision.FaceDetectorOptions(
                    base_options=base_opts,
                    min_detection_confidence=0.3  # 낮은 신뢰도로 설정
                )
                
                detector = vision.FaceDetector.create_from_options(options)
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                result = detector.detect(mp_image)
                detector.close()
                
                if result.detections:
                    detection = result.detections[0]
                    bbox = detection.bounding_box
                    score = detection.categories[0].score if detection.categories else 0
                    
                    # 절대 좌표
                    x1 = bbox.origin_x
                    y1 = bbox.origin_y
                    face_w = bbox.width
                    face_h = bbox.height
                    
                    # 패딩 추가
                    pad = int(0.2 * max(face_w, face_h))
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x1 + face_w + 2 * pad)
                    y2 = min(h, y1 + face_h + 2 * pad)
                    
                    # 정사각형에 가깝게 조정
                    size = max(x2 - x1, y2 - y1)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    x1 = max(0, center_x - size // 2)
                    y1 = max(0, center_y - size // 2)
                    x2 = min(w, x1 + size)
                    y2 = min(h, y1 + size)
                    
                    logger.info(f"✅ Face detected (tasks API): bbox=({x1},{y1},{x2},{y2}), score={score:.3f}")
                    return (int(x1), int(y1), int(x2), int(y2))
                else:
                    logger.warning(f"⚠️ MediaPipe Tasks: 얼굴을 찾지 못함 (frame: {w}x{h})")
                    
            except Exception as e:
                logger.warning(f"⚠️ Tasks API 실패: {e}")
            
            # ========================================
            # 2. solutions API 폴백 (이전 버전)
            # ========================================
            if hasattr(mp, 'solutions'):
                mp_face_detection = mp.solutions.face_detection

                with mp_face_detection.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.3
                ) as face_detection:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)

                    if results.detections:
                        detection = results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        score = detection.score[0] if detection.score else 0

                        x1 = int(bbox.xmin * w)
                        y1 = int(bbox.ymin * h)
                        face_w = int(bbox.width * w)
                        face_h = int(bbox.height * h)

                        pad = int(0.2 * max(face_w, face_h))
                        x1 = max(0, x1 - pad)
                        y1 = max(0, y1 - pad)
                        x2 = min(w, x1 + face_w + 2 * pad)
                        y2 = min(h, y1 + face_h + 2 * pad)

                        size = max(x2 - x1, y2 - y1)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        x1 = max(0, center_x - size // 2)
                        y1 = max(0, center_y - size // 2)
                        x2 = min(w, x1 + size)
                        y2 = min(h, y1 + size)

                        logger.info(f"✅ Face detected (solutions API): bbox=({x1},{y1},{x2},{y2}), score={score:.3f}")
                        return (x1, y1, x2, y2)
                    else:
                        logger.warning(f"⚠️ MediaPipe Solutions: 얼굴을 찾지 못함 (frame: {w}x{h})")
            
            return None

        except ImportError:
            logger.error("❌ MediaPipe not installed!")
            return None
        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _preprocess_face(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional["torch.Tensor"], Optional[Tuple[int, int, int, int]]]:
        """얼굴 영역 추출 및 전처리"""
        import torch
        import cv2

        try:
            # MediaPipe로 얼굴 감지
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options
            from mediapipe.tasks.python.vision.core import vision_task_running_mode, image as mp_image

            # MediaPipe 0.10.0+ tasks API 사용 시도
            if not hasattr(mp, 'solutions'):
                # tasks API 사용
                try:
                    base_opts = base_options.BaseOptions(
                        model_asset_path=None,
                        delegate=base_options.BaseOptions.Delegate.CPU
                    )
                    options = vision.FaceDetectorOptions(
                        base_options=base_opts,
                        running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
                        min_detection_confidence=0.5
                    )
                    face_detector = vision.FaceDetector.create_from_options(options)

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
                    detection_result = face_detector.detect(mp_img)

                    if not detection_result.detections:
                        face_detector.close()
                        return None, None

                    detection = detection_result.detections[0]
                    bbox = detection.bounding_box

                    h, w = frame.shape[:2]
                    x1 = bbox.origin_x
                    y1 = bbox.origin_y
                    x2 = x1 + bbox.width
                    y2 = y1 + bbox.height

                    # 패딩 추가
                    pad = int(0.2 * max(bbox.width, bbox.height))
                    x1 = max(0, x1 - pad)
                    y1 = max(0, y1 - pad)
                    x2 = min(w, x2 + pad)
                    y2 = min(h, y2 + pad)

                    face_region = frame[y1:y2, x1:x2]
                    face_crop = cv2.resize(face_region, (256, 256))

                    face_detector.close()
                    return torch.from_numpy(face_crop).permute(2, 0, 1).unsqueeze(0).float() / 255.0, (x1, y1, x2, y2)

                except Exception as e:
                    logger.debug(f"Face preprocessing (tasks API) failed: {e}, using fallback")
                    return None, None

            # solutions API 사용
            mp_face_detection = mp.solutions.face_detection

            with mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            ) as face_detection:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if not results.detections:
                    return None, None

                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box

                h, w = frame.shape[:2]
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)

                # 패딩 추가
                pad = int(0.2 * max(x2 - x1, y2 - y1))
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)

                face_region = frame[y1:y2, x1:x2]

                # 256x256으로 리사이즈
                face_region = cv2.resize(face_region, (256, 256))

                # 텐서 변환
                face_tensor = torch.from_numpy(face_region).float()
                face_tensor = face_tensor.permute(2, 0, 1) / 255.0
                face_tensor = face_tensor.unsqueeze(0).to(self.device)

                return face_tensor, (x1, y1, x2, y2)

        except Exception as e:
            logger.error(f"Face preprocessing error: {e}")
            return None, None

    def _postprocess_and_blend(
        self,
        original: np.ndarray,
        output: "torch.Tensor",
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """출력을 원본 이미지에 블렌딩"""
        import cv2

        try:
            # 텐서를 numpy로 변환
            output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
            output_np = (output_np * 255).clip(0, 255).astype(np.uint8)

            x1, y1, x2, y2 = bbox
            face_h, face_w = y2 - y1, x2 - x1

            # 원본 크기로 리사이즈
            output_resized = cv2.resize(output_np, (face_w, face_h))

            # 입 부분만 블렌딩 (하단 1/3)
            result = original.copy()
            mouth_y1 = y1 + int(face_h * 0.5)

            # 부드러운 블렌딩을 위한 마스크
            mask = np.zeros((face_h, face_w), dtype=np.float32)
            mask[int(face_h * 0.5):, :] = 1.0

            # 가우시안 블러로 경계 부드럽게
            mask = cv2.GaussianBlur(mask, (21, 21), 10)
            mask = mask[:, :, np.newaxis]

            # 블렌딩
            blended = (
                original[y1:y2, x1:x2].astype(np.float32) * (1 - mask) +
                output_resized.astype(np.float32) * mask
            ).astype(np.uint8)

            result[y1:y2, x1:x2] = blended

            return result

        except Exception as e:
            logger.error(f"Postprocessing error: {e}")
            return original

    def _apply_simple_lipsync(
        self,
        frame: np.ndarray,
        audio: np.ndarray,
    ) -> np.ndarray:
        """
        간단한 립싱크 효과 (모델 없을 때 폴백)

        오디오 에너지에 따라 입 부분 밝기 조절 - 눈에 보이도록 효과 증가
        """
        import cv2

        # 오디오 에너지 계산
        energy = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0

        if energy < 0.01:
            return frame

        # 간단한 밝기 변화로 입 움직임 시뮬레이션
        result = frame.copy()
        h, w = frame.shape[:2]

        # 입 영역 정의 (하단 1/3)
        mouth_top = int(h * 0.55)
        mouth_bottom = int(h * 0.85)
        mouth_left = int(w * 0.25)
        mouth_right = int(w * 0.75)

        mouth_region = result[mouth_top:mouth_bottom, mouth_left:mouth_right].astype(np.float32)

        # 밝기 변화 증가 (30 -> 60)
        brightness_change = energy * 60

        # 입 영역에 밝기 변화 적용
        mouth_region = mouth_region + brightness_change

        # 턱 영역에 미세한 확대/축소 효과 (입 벌림 시뮬레이션)
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
                mouth_region = scaled[start_y:start_y+mouth_h, start_x:start_x+mouth_w].astype(np.float32)

        result[mouth_top:mouth_bottom, mouth_left:mouth_right] = np.clip(
            mouth_region, 0, 255
        ).astype(np.uint8)

        return result

    def cache_face(self, face_id: str, face_data: Dict[str, Any]) -> None:
        """얼굴 데이터 캐시"""
        self._face_cache[face_id] = face_data

    def get_cached_face(self, face_id: str) -> Optional[Dict[str, Any]]:
        """캐시된 얼굴 데이터 반환"""
        return self._face_cache.get(face_id)

    async def cleanup(self) -> None:
        """리소스 정리"""
        # 오디오 버퍼 초기화
        self._audio_buffer.clear()
        
        # 모델 메모리 해제
        self._unet = None
        self._vae = None
        self._audio_processor = None
        self._whisper = None
        self._positional_encoding = None
        self._face_parser = None
        self._face_cache.clear()
        self._initialized = False

        # GPU 메모리 정리
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass

        logger.info("MuseTalk model cleaned up")
    
    def reset_audio_buffer(self) -> None:
        """오디오 버퍼 초기화 (새 세션 시작 시 호출)"""
        self._audio_buffer.clear()
        logger.debug("Audio buffer reset")


class FallbackAudioProcessor:
    """MuseTalk 없을 때 사용하는 폴백 오디오 프로세서"""

    def extract_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        """간단한 오디오 특징 추출"""
        # 에너지 기반 특징
        frame_size = int(sample_rate * 0.025)
        hop_size = int(sample_rate * 0.010)

        features = []
        for i in range(0, max(1, len(audio) - frame_size), hop_size):
            frame = audio[i:i + frame_size]
            energy = np.sqrt(np.mean(frame ** 2))
            zcr = np.sum(np.abs(np.diff(np.sign(frame)))) / (2 * len(frame))
            features.append([energy, zcr] + [0] * 11)

        if not features:
            return np.zeros((1, 13), dtype=np.float32)

        return np.array(features, dtype=np.float32)
