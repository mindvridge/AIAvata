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
                        if self._unet.device:
                            self._positional_encoding.to(self._unet.device)
                        logger.info("PositionalEncoding initialized")
                        
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
                        try:
                            self._face_parser = FaceParsing()
                            logger.info("Face parser initialized")
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

        # 모델이 로드되지 않았으면 간단한 시뮬레이션 사용
        if self._unet is None:
            logger.debug("UNet not loaded, using simple lipsync simulation")
            return self._apply_simple_lipsync(source_frame, audio_chunk)

        try:
            import torch

            # 오디오 특징 추출
            logger.debug(f"Extracting audio features: audio_chunk shape={audio_chunk.shape}, sample_rate={audio_sample_rate}")
            audio_features = self._extract_audio_features(audio_chunk, audio_sample_rate)
            logger.debug(f"Audio features extracted: shape={audio_features.shape}")

            # VAE가 없으면 fallback
            if self._vae is None:
                logger.warning("VAE not available, using simple fallback")
                return self._apply_simple_lipsync(source_frame, audio_chunk)
            
            # 얼굴 영역 추출 (전체 프레임 사용, 256x256으로 리사이즈)
            import cv2
            face_crop = cv2.resize(source_frame, (256, 256))
            
            # VAE로 얼굴 이미지를 latent로 인코딩
            # get_latents_for_unet은 이미지 경로나 numpy array를 받을 수 있음
            latent_input = self._vae.get_latents_for_unet(face_crop)
            latent_input = latent_input.to(self.device)
            if self.fp16:
                latent_input = latent_input.half()

            # 오디오 특징에 PositionalEncoding 적용
            # 실제 MuseTalk에서는 Whisper encoder를 통해 처리되지만,
            # 실시간 처리를 위해 간단한 형태로 변환
            if self._positional_encoding:
                # audio_features는 [batch, seq_len, features] 형태여야 함
                # 현재는 [batch, features, seq_len] 또는 다른 형태일 수 있음
                logger.debug(f"Audio features before PE: shape={audio_features.shape}")
                
                # 형태 변환: [batch, seq_len, features] 형태로 맞춤
                if len(audio_features.shape) == 2:
                    audio_features = audio_features.unsqueeze(0)  # [1, seq_len, features]
                elif len(audio_features.shape) == 4:  # [batch, channels, height, width]
                    # Whisper feature extractor 출력 형태
                    b, c, h, w = audio_features.shape
                    audio_features = audio_features.view(b, h, c * w)  # [batch, seq_len, features]
                
                # PositionalEncoding 적용
                audio_features = self._positional_encoding(audio_features)
                logger.debug(f"Audio features after PE: shape={audio_features.shape}")
                
                # 형태 변환: UNet이 기대하는 형태로 변환
                # UNet은 [batch, seq_len, features] 형태를 기대하지만,
                # 실제로는 [batch, seq_len, hidden_dim] 형태여야 함
                # MuseTalk에서는 Whisper encoder의 hidden states를 사용
                # 여기서는 간단하게 형태를 맞춤
                if len(audio_features.shape) == 3:
                    # [batch, seq_len, features] 형태 유지
                    pass
                else:
                    logger.warning(f"Unexpected audio features shape: {audio_features.shape}")
                    # 형태 조정 시도
                    if len(audio_features.shape) == 2:
                        audio_features = audio_features.unsqueeze(0)
            else:
                logger.warning("PositionalEncoding not available, skipping")

            # 추론
            with torch.no_grad():
                # UNet 추론 (실제 MuseTalk 방식)
                timesteps = torch.tensor([0], device=self.device)
                
                # UNet 호출 방식: latent_input과 audio_features 결합
                # audio_features는 [batch, seq_len, hidden_dim] 형태여야 함
                logger.debug(f"UNet input - latent_input: {latent_input.shape}, audio_features: {audio_features.shape}")
                
                try:
                    pred_latents = self._unet.model(
                        latent_input,
                        timesteps,
                        encoder_hidden_states=audio_features
                    ).sample
                    logger.debug(f"UNet output: {pred_latents.shape}")
                except Exception as e:
                    logger.error(f"UNet inference failed: {e}", exc_info=True)
                    raise
                
                # VAE 디코딩
                pred_latents = pred_latents.to(dtype=self._vae.vae.dtype)
                recon = self._vae.decode_latents(pred_latents)
                
                # 첫 번째 프레임만 사용 (배치 크기 1)
                if isinstance(recon, (list, tuple)):
                    output = recon[0]
                elif len(recon.shape) == 4 and recon.shape[0] > 1:
                    output = recon[0:1]  # 첫 번째만
                else:
                    output = recon

            # 후처리: VAE 출력은 BGR 이미지이므로 그대로 사용
            # output은 numpy array [H, W, 3] 형태
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
            else:
                output_np = output
            
            # 원본 크기로 리사이즈
            h, w = source_frame.shape[:2]
            result_frame = cv2.resize(output_np, (w, h))
            
            return result_frame

        except Exception as e:
            logger.error(f"MuseTalk inference error: {e}", exc_info=True)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # 오류 발생 시 원본 프레임 반환
            return source_frame

    def _extract_audio_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> "torch.Tensor":
        """오디오에서 특징 추출 (실제 MuseTalk AudioProcessor 사용)"""
        import torch

        if self._audio_processor is not None:
            try:
                # AudioProcessor의 feature_extractor를 직접 사용
                # 실시간 오디오 청크를 처리하기 위해 librosa로 리샘플링 후 추출
                import librosa
                
                # 16000 Hz로 리샘플링 (MuseTalk 요구사항)
                if sample_rate != 16000:
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                
                # feature_extractor 사용 (Whisper feature extractor)
                audio_feature = self._audio_processor.feature_extractor(
                    audio,
                    return_tensors="pt",
                    sampling_rate=16000
                ).input_features
                
                # 디바이스로 이동
                audio_feature = audio_feature.to(self.device)
                if self.fp16:
                    audio_feature = audio_feature.half()
                
                return audio_feature
            except Exception as e:
                logger.warning(f"AudioProcessor feature extraction failed: {e}, using fallback")
                # 폴백: 간단한 특징 추출
                features = self._simple_audio_features(audio, sample_rate)
                return torch.from_numpy(features).to(self.device).unsqueeze(0)
        else:
            # 간단한 MFCC 추출
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

        오디오 에너지에 따라 입 부분 밝기 조절
        """
        # 오디오 에너지 계산
        energy = np.sqrt(np.mean(audio ** 2)) if len(audio) > 0 else 0

        if energy < 0.01:
            return frame

        # 간단한 밝기 변화로 입 움직임 시뮬레이션
        result = frame.copy()

        # 하단 1/3 영역에 약간의 밝기 변화
        h, w = frame.shape[:2]
        mouth_region = result[int(h * 0.6):int(h * 0.85), int(w * 0.3):int(w * 0.7)]

        brightness_change = int(energy * 30)
        mouth_region = np.clip(
            mouth_region.astype(np.int16) + brightness_change,
            0, 255
        ).astype(np.uint8)

        result[int(h * 0.6):int(h * 0.85), int(w * 0.3):int(w * 0.7)] = mouth_region

        return result

    def cache_face(self, face_id: str, face_data: Dict[str, Any]) -> None:
        """얼굴 데이터 캐시"""
        self._face_cache[face_id] = face_data

    def get_cached_face(self, face_id: str) -> Optional[Dict[str, Any]]:
        """캐시된 얼굴 데이터 반환"""
        return self._face_cache.get(face_id)

    async def cleanup(self) -> None:
        """리소스 정리"""
        self._unet = None
        self._vae = None
        self._audio_processor = None
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
