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
    ):
        """
        Initialize MuseTalk Model.

        Args:
            model_dir: 모델 파일 디렉토리
            device: 연산 디바이스
            fp16: FP16 추론 사용 여부
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.fp16 = fp16

        # 모델 컴포넌트
        self._audio_processor = None
        self._unet = None
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
                                logger.info("Face parser initialized")
                            else:
                                logger.warning("⚠️ Face parser 모델 없음 (79999_iter.pth)")
                                logger.warning("   다운로드: https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth")
                                logger.warning("   저장 경로: models/face-parse-bisent/79999_iter.pth")
                                logger.warning("   (립싱크는 작동하지만 품질이 저하될 수 있습니다)")
                                self._face_parser = None
                        except Exception as e:
                            logger.warning(f"Face parser initialization failed (non-critical): {e}")
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

        # 모델이 로드되지 않았으면 에러 표시
        if self._unet is None:
            logger.error("❌ MuseTalk UNet 모델이 로드되지 않았습니다!")
            logger.error("   필요한 파일: models/musetalk/musetalkV15/unet.pth")
            logger.error("   해결방법: run.bat를 다시 실행하거나 수동으로 다운로드하세요.")
            return source_frame

        try:
            import torch

            # 오디오 특징 추출
            logger.info(f"🎤 Extracting audio features: audio_chunk shape={audio_chunk.shape}, sample_rate={audio_sample_rate}")
            audio_features = self._extract_audio_features(audio_chunk, audio_sample_rate)
            logger.info(f"🎤 Audio features extracted: shape={audio_features.shape}")

            # VAE가 없으면 에러 표시
            if self._vae is None:
                logger.error("❌ MuseTalk VAE 모델이 로드되지 않았습니다!")
                logger.error("   필요한 파일: models/musetalk/sd-vae-ft-mse/")
                logger.error("   해결방법: run.bat를 다시 실행하거나 수동으로 다운로드하세요.")
                return source_frame
            
            # 얼굴 영역 추출 (MediaPipe로 얼굴 감지 후 크롭)
            import cv2

            # MediaPipe로 얼굴 위치 찾기
            face_bbox = self._detect_face_bbox(source_frame)

            if face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                # 얼굴 영역만 크롭
                face_region = source_frame[y1:y2, x1:x2]
                face_crop = cv2.resize(face_region, (256, 256))
                logger.debug(f"Face detected: bbox=({x1}, {y1}, {x2}, {y2})")
            else:
                # 얼굴 감지 실패 시 전체 프레임 사용
                face_crop = cv2.resize(source_frame, (256, 256))
                face_bbox = None
                logger.debug("Face not detected, using full frame")
            
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
            device_obj = torch.device(self.device if torch.cuda.is_available() else "cpu")
            timesteps = torch.tensor([0], device=device_obj)
            
            # audio_features 형태에 따라 처리
            logger.debug(f"Audio features shape before processing: {audio_features.shape}")
            
            # MuseTalk의 get_whisper_chunk 결과는 [T, (c h) w] 형태
            # 여기서는 실시간으로 [1, seq_len, features] 형태가 나오는데
            # 이를 MuseTalk의 형식인 [batch, (c h) w]로 변환해야 함
            
            # audio_features가 [batch, features] 형태인 경우
            if len(audio_features.shape) == 2:
                # [batch, features] -> [batch, 1, features] -> [batch, (1) features]
                # MuseTalk에서는 여러 프레임의 오디오를 쌓지만, 실시간에서는 1프레임씩
                # PositionalEncoding을 위해 [batch, seq_len, features] 형태로 변환
                if audio_features.shape[1] == 384:  # 단일 feature vector
                    # [batch, 384] -> [batch, 1, 384]
                    audio_features = audio_features.unsqueeze(1)
                else:
                    # 이미 올바른 형태
                    pass
            elif len(audio_features.shape) == 3:
                # [batch, seq_len, features] 형태
                pass
            elif len(audio_features.shape) == 4:
                # [batch, channels, height, width] -> [batch, (c h) w] (MuseTalk 형식)
                from einops import rearrange
                audio_features = rearrange(audio_features, 'b c h w -> b (c h) w')
            
            # PositionalEncoding 적용 (MuseTalk 방식)
            # MuseTalk의 get_whisper_chunk 결과는 [batch, (c h) w] 형태
            # PositionalEncoding은 [batch, seq_len, d_model] 형태를 기대
            if self._positional_encoding:
                # audio_features 형태에 따라 처리
                if len(audio_features.shape) == 2:
                    # [batch, features] -> [batch, 1, features]
                    audio_features = audio_features.unsqueeze(1)
                
                # MuseTalk에서는 [batch, (c h) w] 형태인데, 이것을 PositionalEncoding에 전달
                # PositionalEncoding은 [batch, seq_len, d_model] 형태를 기대
                # 여기서 seq_len은 (c h)이고, d_model은 w (384)
                if len(audio_features.shape) == 3:
                    # [batch, seq_len, features] 형태
                    # PositionalEncoding 적용
                    audio_features = self._positional_encoding(audio_features.to(device_obj))
                    logger.debug(f"Audio features after PE: shape={audio_features.shape}")
                elif len(audio_features.shape) == 2:
                    # [batch, features] 형태 -> [batch, 1, features]
                    audio_features = audio_features.unsqueeze(1)
                    audio_features = self._positional_encoding(audio_features.to(device_obj))
                    logger.debug(f"Audio features after PE: shape={audio_features.shape}")
            
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
                    # UNet2DConditionModel은 BaseOutput 객체를 반환하며 .sample 속성을 가짐
                    logger.info(f"🔄 UNet inference starting - latent: {latent_input.shape}, audio: {audio_features.shape}")
                    unet_output = self._unet.model(
                        latent_input,
                        timesteps,
                        encoder_hidden_states=audio_features
                    )

                    # .sample 속성 접근 (UNet2DConditionModel의 반환값)
                    pred_latents = unet_output.sample

                    logger.info(f"✅ UNet output: {pred_latents.shape}")
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
                logger.info(f"🔄 VAE decoding starting... (VAE device={vae_device}, dtype={vae_dtype})")
                pred_latents = pred_latents.to(dtype=self._vae.vae.dtype)
                recon = self._vae.decode_latents(pred_latents)
                logger.info(f"✅ VAE decoded: type={type(recon)}, shape={recon.shape if hasattr(recon, 'shape') else 'N/A'}")
                
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
                # 원본 크기로 리사이즈
                result_256 = cv2.resize(output_np, (256, 256), interpolation=cv2.INTER_LINEAR)

                # 얼굴 크롭 이미지 사용 (전체 프레임이 아닌 face_crop 사용!)
                # 버그 수정: source_frame 대신 face_crop 사용해야 함
                source_256 = face_crop.copy()

                # 🔑 핵심: Face Parser로 입 영역 마스크 생성
                mask = None
                if self._face_parser is not None:
                    try:
                        # Face Parser로 정확한 입 영역 마스크 생성
                        # FaceParsing은 PIL Image를 반환하므로 numpy로 변환 필요
                        from PIL import Image
                        parsing_result = self._face_parser(source_256)

                        # 반환값 타입 확인 (디버깅)
                        logger.debug(f"Face parser result type: {type(parsing_result)}")

                        # int, None 등 유효하지 않은 반환값 처리
                        if parsing_result is None or isinstance(parsing_result, (int, float, bool)):
                            logger.debug(f"Face parser returned invalid type: {type(parsing_result)}")
                            parsing_result = None

                        if parsing_result is not None:
                            # PIL Image → numpy array 변환
                            if isinstance(parsing_result, Image.Image):
                                parsing_array = np.array(parsing_result)
                            elif isinstance(parsing_result, np.ndarray):
                                parsing_array = parsing_result
                            elif hasattr(parsing_result, '__array__'):
                                parsing_array = np.array(parsing_result)
                            else:
                                logger.debug(f"Unknown parsing result type: {type(parsing_result)}")
                                parsing_array = None

                            # parsing_array가 유효한 경우에만 처리
                            if parsing_array is not None and hasattr(parsing_array, 'shape') and len(parsing_array.shape) >= 2:
                                # MuseTalk face parsing labels (raw mode):
                                # 실제 반환값은 255 (face) vs 0 (background)
                                # 입 영역만 추출하려면 하단 영역 마스크 생성
                                lip_mask = np.zeros((256, 256), dtype=np.float32)
                                # 얼굴 영역 (255) 확인
                                face_mask = (parsing_array > 128).astype(np.float32)

                                if np.sum(face_mask) > 1000:  # 얼굴이 감지된 경우
                                    # 입 영역: 얼굴의 하단 50% 영역
                                    h_mask = parsing_array.shape[0]
                                    lip_region_start = int(h_mask * 0.5)
                                    lip_mask[lip_region_start:, :] = face_mask[lip_region_start:, :]

                                    # 마스크 확장 및 블러 (블러 감소로 선명한 효과)
                                    if np.sum(lip_mask) > 100:
                                        kernel = np.ones((3, 3), np.uint8)
                                        lip_mask = cv2.dilate(lip_mask, kernel, iterations=1)
                                        lip_mask = cv2.GaussianBlur(lip_mask, (11, 11), 0)
                                        mask = lip_mask
                                        logger.debug(f"Face parser mask created: {np.sum(mask > 0)} pixels")
                    except Exception as e:
                        logger.debug(f"Face parser mask skipped (using ellipse fallback): {e}")
                        mask = None

                # Face Parser 실패 시 간단한 타원형 마스크 사용
                if mask is None:
                    mask = np.zeros((256, 256), dtype=np.float32)
                    # 입 위치 추정 (얼굴 크롭 기준)
                    # 256x256 얼굴 크롭에서 입은 보통 y=160~210 영역 (턱 포함)
                    # 마스크를 더 작고 정확한 입 영역에 집중
                    center_x, center_y = 128, 185  # 입 중심 (약간 아래로)
                    axes = (45, 30)  # 타원 크기 축소 - 입 영역만 정확히
                    cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 1.0, -1)
                    # 가우시안 블러를 더 줄여서 선명한 효과
                    mask = cv2.GaussianBlur(mask, (11, 11), 0)
                    logger.debug(f"Using ellipse mask: center=({center_x},{center_y}), axes={axes}")

                # 3채널로 확장
                mask_3ch = np.stack([mask, mask, mask], axis=-1)

                # 블렌딩 강도 조절 (0.0~1.0, 높을수록 립싱크 효과 강함)
                blend_alpha = 0.85  # 85% 립싱크 결과 사용

                # VAE 출력과 원본의 차이 로깅 (디버그용)
                if logger.isEnabledFor(logging.DEBUG):
                    diff = np.abs(result_256.astype(np.float32) - source_256.astype(np.float32))
                    mouth_region_diff = diff[155:215, 83:173]  # 입 영역만
                    avg_diff = np.mean(mouth_region_diff)
                    logger.debug(f"VAE output mouth region diff from source: avg={avg_diff:.1f}")

                # 블렌딩: 원본 * (1-mask*alpha) + 결과 * (mask*alpha)
                effective_mask = mask_3ch * blend_alpha
                blended_256 = (source_256.astype(np.float32) * (1 - effective_mask) +
                               result_256.astype(np.float32) * effective_mask)
                blended_256 = np.clip(blended_256, 0, 255).astype(np.uint8)

                # 얼굴 bbox가 있으면 해당 영역에만 결과 적용
                if face_bbox is not None:
                    x1, y1, x2, y2 = face_bbox
                    face_w, face_h = x2 - x1, y2 - y1

                    # 결과를 원본 얼굴 크기로 리사이즈
                    result_face = cv2.resize(blended_256, (face_w, face_h), interpolation=cv2.INTER_LINEAR)

                    # 원본 프레임에 결과 페이스트
                    result_frame = source_frame.copy()
                    result_frame[y1:y2, x1:x2] = result_face

                    logger.info(f"✅ MuseTalk lip sync SUCCESS: face bbox=({x1},{y1},{x2},{y2})")
                    return result_frame
                else:
                    # 얼굴 감지 실패 시 전체 프레임 리사이즈
                    result_frame = cv2.resize(blended_256, (w, h), interpolation=cv2.INTER_LINEAR)

                    if result_frame is not None and result_frame.shape[:2] == (h, w):
                        logger.info(f"✅ MuseTalk lip sync SUCCESS: output shape={result_frame.shape}")
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
        
        Args:
            audio: 오디오 배열 (float32, [-1, 1])
            sample_rate: 오디오 샘플레이트
            
        Returns:
            Whisper encoder hidden states (torch.Tensor)
        """
        import torch

        if self._audio_processor is None:
            # 폴백: 간단한 특징 추출
            features = self._simple_audio_features(audio, sample_rate)
            return torch.from_numpy(features).to(self.device).unsqueeze(0)

        try:
            import librosa
            
            # 16000 Hz로 리샘플링 (MuseTalk 요구사항)
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
            
            # 실시간 처리를 위한 오디오 버퍼링
            # MuseTalk은 여러 프레임의 오디오를 함께 처리하므로, 버퍼 유지 필요
            self._audio_buffer.append(audio.copy())
            
            # 버퍼 크기 제한 (메모리 관리)
            total_length = sum(len(chunk) for chunk in self._audio_buffer)
            while total_length > self._audio_buffer_size:
                if len(self._audio_buffer) > 0:
                    removed = self._audio_buffer.pop(0)
                    total_length -= len(removed)
            
            # 버퍼 결합 (최근 오디오들 사용)
            if len(self._audio_buffer) > 1:
                audio_combined = np.concatenate(self._audio_buffer)
            else:
                audio_combined = audio
            
            # 최소 길이 보장 (Whisper 요구사항)
            min_length = 16000 * 0.5  # 최소 0.5초
            if len(audio_combined) < min_length:
                # 패딩 추가
                padding_length = int(min_length - len(audio_combined))
                audio_combined = np.pad(audio_combined, (0, padding_length), mode='constant', constant_values=0)
            
            # Whisper feature extractor로 mel spectrogram 추출
            device_obj = torch.device(self.device if torch.cuda.is_available() else "cpu")
            audio_feature = self._audio_processor.feature_extractor(
                audio_combined,
                return_tensors="pt",
                sampling_rate=16000
            ).input_features
            
            audio_feature = audio_feature.to(device_obj)
            if self._weight_dtype:
                audio_feature = audio_feature.to(dtype=self._weight_dtype)
            
            # Whisper encoder로 hidden states 추출 (실제 MuseTalk 방식)
            if self._whisper is not None:
                with torch.no_grad():
                    audio_feats = self._whisper.encoder(
                        audio_feature, 
                        output_hidden_states=True
                    ).hidden_states
                    
                    # 모든 레이어의 hidden states를 스택 (MuseTalk 방식)
                    audio_feats = torch.stack(audio_feats, dim=2)  # [batch, seq_len, num_layers, hidden_dim]
                    
                    # MuseTalk 형식으로 변환
                    # get_whisper_chunk에서는 더 복잡한 처리를 하지만,
                    # 실시간 처리를 위해 간단히 처리
                    b, seq_len, num_layers, hidden_dim = audio_feats.shape
                    
                    # 실시간 처리: 마지막 타임스텝만 사용하거나 평균 사용
                    if seq_len > 1:
                        # 여러 타임스텝이 있으면 평균 사용
                        audio_feats = audio_feats.mean(dim=1, keepdim=True)  # [batch, 1, num_layers, hidden_dim]
                    
                    # 형태 조정: [batch, 1, num_layers, hidden_dim] -> [batch, num_layers, hidden_dim]
                    audio_feats = audio_feats.squeeze(1)  # [batch, num_layers, hidden_dim]
                    
                    # MuseTalk 방식: 모든 레이어의 hidden states를 스택
                    # [batch, seq_len, num_layers, hidden_dim]
                    # MuseTalk의 get_whisper_chunk에서는 이 형태를 [batch, (c h) w]로 변환
                    # 여기서는 실시간 처리를 위해 간단화
                    # 실제로는 여러 프레임의 오디오를 버퍼링하여 처리해야 하지만,
                    # 여기서는 마지막 레이어의 hidden state만 사용 (384 차원)
                    # 또는 모든 레이어의 평균 사용
                    
                    # 마지막 레이어만 사용 (hidden_dim = 384)
                    audio_feats_last = audio_feats[:, -1, :]  # [batch, hidden_dim]
                    
                    # 또는 모든 레이어의 평균 사용
                    # audio_feats_last = audio_feats.mean(dim=1)  # [batch, hidden_dim]
                    
                    # PositionalEncoding을 위한 형태: [batch, 1, hidden_dim]
                    audio_feats_last = audio_feats_last.unsqueeze(1)  # [batch, 1, 384]
                    
                    return audio_feats_last
            else:
                # Whisper 없으면 feature_extractor 출력만 사용
                return audio_feature
                
        except Exception as e:
            logger.warning(f"Audio feature extraction failed: {e}, using fallback")
            import traceback
            logger.debug(traceback.format_exc())
            # 폴백: 간단한 특징 추출
            features = self._simple_audio_features(audio, sample_rate)
            return torch.from_numpy(features).to(self.device).unsqueeze(0)

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
        MediaPipe로 얼굴 영역 감지

        Args:
            frame: 입력 프레임 (BGR)

        Returns:
            (x1, y1, x2, y2) 얼굴 bounding box 또는 None
        """
        import cv2

        try:
            import mediapipe as mp

            mp_face_detection = mp.solutions.face_detection

            with mp_face_detection.FaceDetection(
                model_selection=1,  # 0: 2m 이내, 1: 5m 이내
                min_detection_confidence=0.5
            ) as face_detection:
                h, w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)

                if not results.detections:
                    return None

                # 가장 큰 얼굴 선택 (여러 얼굴이 있을 경우)
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box

                # 상대 좌표를 절대 좌표로 변환
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                face_w = int(bbox.width * w)
                face_h = int(bbox.height * h)

                # 패딩 추가 (얼굴 주변 여유 공간)
                pad = int(0.3 * max(face_w, face_h))
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

                return (x1, y1, x2, y2)

        except ImportError:
            logger.debug("MediaPipe not installed, face detection unavailable")
            return None
        except Exception as e:
            logger.debug(f"Face detection failed: {e}")
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
