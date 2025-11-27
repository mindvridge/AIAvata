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
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np

logger = logging.getLogger(__name__)

# MuseTalk 모델 경로
MUSETALK_MODEL_DIR = os.getenv("MUSETALK_MODEL_DIR", "models/musetalk")


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
                # MuseTalk는 별도 설치가 필요함
                from musetalk.audio import AudioProcessor
                from musetalk.models.unet import MuseTalkUNet
                from musetalk.models.vae import VAEModel

                # 오디오 프로세서
                self._audio_processor = AudioProcessor()

                # UNet 모델 로드
                unet_path = self.model_dir / "musetalk.pth"
                if unet_path.exists():
                    self._unet = MuseTalkUNet()
                    self._unet.load_state_dict(torch.load(unet_path, map_location=self.device))
                    self._unet.to(self.device)
                    self._unet.eval()

                    if self.fp16:
                        self._unet = self._unet.half()

                # VAE 모델 로드
                vae_path = self.model_dir / "vae.pth"
                if vae_path.exists():
                    self._vae = VAEModel()
                    self._vae.load_state_dict(torch.load(vae_path, map_location=self.device))
                    self._vae.to(self.device)
                    self._vae.eval()

                logger.info("MuseTalk models loaded successfully")

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

        # 모델이 로드되지 않았으면 원본 반환
        if self._unet is None:
            return self._apply_simple_lipsync(source_frame, audio_chunk)

        try:
            import torch

            # 오디오 특징 추출
            audio_features = self._extract_audio_features(audio_chunk, audio_sample_rate)

            # 얼굴 영역 추출 및 전처리
            face_tensor, face_bbox = self._preprocess_face(source_frame)

            if face_tensor is None:
                return source_frame

            # 추론
            with torch.no_grad():
                if self.fp16:
                    face_tensor = face_tensor.half()
                    audio_features = audio_features.half()

                # UNet을 통한 립싱크 생성
                output = self._unet(face_tensor, audio_features)

                # VAE 디코딩
                if self._vae is not None:
                    output = self._vae.decode(output)

            # 후처리 및 원본에 블렌딩
            result_frame = self._postprocess_and_blend(
                source_frame, output, face_bbox
            )

            return result_frame

        except Exception as e:
            logger.error(f"MuseTalk inference error: {e}")
            return source_frame

    def _extract_audio_features(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> "torch.Tensor":
        """오디오에서 특징 추출"""
        import torch

        if self._audio_processor is not None:
            features = self._audio_processor.extract_features(audio, sample_rate)
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
