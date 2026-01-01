"""
TensorRT utilities for model acceleration.

TensorRT 모델 최적화 및 추론 유틸리티
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import hashlib
import pickle

import torch
import numpy as np

logger = logging.getLogger(__name__)


def get_model_hash(model: torch.nn.Module, sample_inputs: Tuple[Any, ...]) -> str:
    """
    모델 구조와 입력 형태를 기반으로 해시값 생성
    
    Args:
        model: PyTorch 모델
        sample_inputs: 샘플 입력 튜플
        
    Returns:
        해시 문자열
    """
    # 모델 파라미터 개수와 구조 기반 해시
    param_count = sum(p.numel() for p in model.parameters())
    input_shapes = [tuple(inp.shape) if hasattr(inp, 'shape') else str(type(inp)) for inp in sample_inputs]
    
    hash_str = f"{param_count}_{input_shapes}"
    return hashlib.md5(hash_str.encode()).hexdigest()[:12]


def convert_unet_to_tensorrt(
    unet_model: torch.nn.Module,
    sample_latent: torch.Tensor,
    sample_timesteps: torch.Tensor,
    sample_encoder_hidden_states: torch.Tensor,
    engine_dir: Path,
    fp16: bool = True,
    workspace_size: int = 1024 * 1024 * 1024,  # 1GB
) -> Optional[torch.nn.Module]:
    """
    UNet 모델을 TensorRT로 변환
    
    Args:
        unet_model: PyTorch UNet 모델
        sample_latent: 샘플 latent 입력 [batch, channels, height, width]
        sample_timesteps: 샘플 timesteps [batch]
        sample_encoder_hidden_states: 샘플 encoder hidden states [batch, seq_len, hidden_dim]
        engine_dir: TensorRT 엔진 저장 디렉토리
        fp16: FP16 정밀도 사용 여부
        workspace_size: TensorRT 워크스페이스 크기 (바이트)
        
    Returns:
        TensorRT로 변환된 모델 또는 None (변환 실패 시)
    """
    try:
        import torch_tensorrt
        logger.info("torch-tensorrt found, attempting conversion...")
    except (ImportError, OSError) as e:
        # ImportError: 패키지가 설치되지 않음
        # OSError: Windows에서 DLL 로드 실패 (torch-tensorrt의 알려진 문제)
        logger.warning(f"torch-tensorrt not available: {e}")
        logger.info("Falling back to PyTorch inference (this is normal on Windows)")
        return None
    
    engine_dir.mkdir(parents=True, exist_ok=True)
    
    # 엔진 파일 이름 생성 (입력 형태 기반)
    model_hash = get_model_hash(
        unet_model,
        (sample_latent, sample_timesteps, sample_encoder_hidden_states)
    )
    engine_name = f"unet_{model_hash}_{'fp16' if fp16 else 'fp32'}.trt"
    engine_path = engine_dir / engine_name
    
    # 이미 생성된 엔진이 있으면 로드
    if engine_path.exists():
        logger.info(f"Loading existing TensorRT engine: {engine_path}")
        try:
            # torch-tensorrt는 엔진을 직접 로드할 수 없으므로 재변환 필요
            # 대신 torch.jit.trace된 모델을 사용하거나 ONNX를 거쳐야 함
            logger.info("TensorRT engine found, but torch-tensorrt requires recompilation")
        except Exception as e:
            logger.warning(f"Failed to load TensorRT engine: {e}")
    
    # 모델을 eval 모드로 설정
    unet_model.eval()
    
    try:
        logger.info("Converting UNet to TensorRT...")
        logger.info(f"  Input shapes: latent={sample_latent.shape}, timesteps={sample_timesteps.shape}, encoder={sample_encoder_hidden_states.shape}")
        logger.info(f"  Precision: {'FP16' if fp16 else 'FP32'}")
        logger.info(f"  Workspace size: {workspace_size / (1024**2):.1f} MB")
        
        # torch-tensorrt 변환 설정
        compile_spec = {
            "inputs": [
                torch_tensorrt.Input(
                    shape=sample_latent.shape,
                    dtype=torch.float16 if fp16 else torch.float32,
                ),
                torch_tensorrt.Input(
                    shape=sample_timesteps.shape,
                    dtype=sample_timesteps.dtype,
                ),
                torch_tensorrt.Input(
                    shape=sample_encoder_hidden_states.shape,
                    dtype=torch.float16 if fp16 else torch.float32,
                ),
            ],
            "enabled_precisions": {torch.float, torch.half} if fp16 else {torch.float},
            "workspace_size": workspace_size,
        }
        
        # torch.jit.trace로 먼저 변환
        # UNet2DConditionModel은 encoder_hidden_states를 keyword argument로 받지만,
        # trace를 위해서는 wrapper 함수 필요
        logger.info("Tracing UNet model with torch.jit.trace...")
        
        # Wrapper 함수로 keyword argument를 positional로 변환
        class UNetWrapper(torch.nn.Module):
            def __init__(self, unet):
                super().__init__()
                self.unet = unet
            
            def forward(self, sample, timestep, encoder_hidden_states):
                return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states)
        
        wrapped_unet = UNetWrapper(unet_model)
        wrapped_unet.eval()
        
        with torch.no_grad():
            traced_model = torch.jit.trace(
                wrapped_unet,
                (sample_latent, sample_timesteps, sample_encoder_hidden_states),
                strict=False,
            )
        
        # TensorRT로 컴파일
        logger.info("Compiling UNet with TensorRT (this may take several minutes)...")
        trt_model = torch_tensorrt.compile(
            traced_model,
            **compile_spec
        )
        
        logger.info("✅ UNet TensorRT conversion successful!")
        
        # 엔진 정보 저장 (실제 엔진은 torch-tensorrt 내부에 저장됨)
        engine_info = {
            "model_hash": model_hash,
            "input_shapes": {
                "latent": list(sample_latent.shape),
                "timesteps": list(sample_timesteps.shape),
                "encoder_hidden_states": list(sample_encoder_hidden_states.shape),
            },
            "fp16": fp16,
        }
        info_path = engine_dir / f"{engine_name}.info"
        with open(info_path, "wb") as f:
            pickle.dump(engine_info, f)
        
        return trt_model
        
    except Exception as e:
        logger.error(f"Failed to convert UNet to TensorRT: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Falling back to PyTorch inference")
        return None


def convert_vae_decoder_to_tensorrt(
    vae_decoder: torch.nn.Module,
    sample_latents: torch.Tensor,
    engine_dir: Path,
    fp16: bool = True,
    workspace_size: int = 1024 * 1024 * 1024,  # 1GB
) -> Optional[torch.nn.Module]:
    """
    VAE 디코더를 TensorRT로 변환
    
    Args:
        vae_decoder: PyTorch VAE 디코더 모델
        sample_latents: 샘플 latent 입력 [batch, channels, height, width]
        engine_dir: TensorRT 엔진 저장 디렉토리
        fp16: FP16 정밀도 사용 여부
        workspace_size: TensorRT 워크스페이스 크기 (바이트)
        
    Returns:
        TensorRT로 변환된 모델 또는 None (변환 실패 시)
    """
    try:
        import torch_tensorrt
        logger.info("torch-tensorrt found, attempting VAE decoder conversion...")
    except ImportError:
        logger.warning("torch-tensorrt not installed. Install with: pip install torch-tensorrt")
        logger.warning("Falling back to PyTorch inference")
        return None
    
    engine_dir.mkdir(parents=True, exist_ok=True)
    
    # 엔진 파일 이름 생성
    model_hash = get_model_hash(vae_decoder, (sample_latents,))
    engine_name = f"vae_decoder_{model_hash}_{'fp16' if fp16 else 'fp32'}.trt"
    engine_path = engine_dir / engine_name
    
    # 모델을 eval 모드로 설정
    vae_decoder.eval()
    
    try:
        logger.info("Converting VAE decoder to TensorRT...")
        logger.info(f"  Input shape: {sample_latents.shape}")
        logger.info(f"  Precision: {'FP16' if fp16 else 'FP32'}")
        
        # torch-tensorrt 변환 설정
        compile_spec = {
            "inputs": [
                torch_tensorrt.Input(
                    shape=sample_latents.shape,
                    dtype=torch.float16 if fp16 else torch.float32,
                ),
            ],
            "enabled_precisions": {torch.float, torch.half} if fp16 else {torch.float},
            "workspace_size": workspace_size,
        }
        
        # torch.jit.trace
        logger.info("Tracing VAE decoder model...")
        with torch.no_grad():
            traced_model = torch.jit.trace(
                vae_decoder,
                (sample_latents,),
                strict=False,
            )
        
        # TensorRT로 컴파일
        logger.info("Compiling VAE decoder with TensorRT...")
        trt_model = torch_tensorrt.compile(
            traced_model,
            **compile_spec
        )
        
        logger.info("✅ VAE decoder TensorRT conversion successful!")
        
        # 엔진 정보 저장
        engine_info = {
            "model_hash": model_hash,
            "input_shape": list(sample_latents.shape),
            "fp16": fp16,
        }
        info_path = engine_dir / f"{engine_name}.info"
        with open(info_path, "wb") as f:
            pickle.dump(engine_info, f)
        
        return trt_model
        
    except Exception as e:
        logger.error(f"Failed to convert VAE decoder to TensorRT: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Falling back to PyTorch inference")
        return None


def check_tensorrt_available() -> bool:
    """
    TensorRT가 사용 가능한지 확인
    
    Returns:
        TensorRT 사용 가능 여부
    """
    try:
        import torch_tensorrt
        if torch.cuda.is_available():
            logger.info("✅ TensorRT (torch-tensorrt) is available")
            return True
        else:
            logger.warning("CUDA not available, TensorRT cannot be used")
            return False
    except ImportError:
        logger.debug("torch-tensorrt not installed")
        return False
