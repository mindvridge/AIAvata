"""
VAE 모델 다운로드 스크립트

Stable Diffusion VAE 모델을 Hugging Face에서 다운로드합니다.
MuseTalk 립싱크에 필수입니다.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def download_vae_model():
    """VAE 모델 다운로드"""
    
    logger.info("=" * 60)
    logger.info("VAE 모델 다운로드 시작")
    logger.info("=" * 60)
    logger.info("")
    
    # 모델 경로
    vae_model_dir = Path("models/sd-vae-ft-mse")
    
    # 이미 존재하는지 확인
    if vae_model_dir.exists():
        logger.info(f"✅ VAE 모델이 이미 존재합니다: {vae_model_dir}")
        logger.info("다시 다운로드하려면 기존 디렉토리를 삭제하세요.")
        return True
    
    logger.info("[1/2] 모델 디렉토리 생성...")
    vae_model_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ 디렉토리 생성: {vae_model_dir}")
    logger.info("")
    
    logger.info("[2/2] Hugging Face에서 VAE 모델 다운로드 중...")
    logger.info("이 과정은 시간이 오래 걸릴 수 있습니다 (약 335MB)...")
    logger.info("모델: stabilityai/sd-vae-ft-mse")
    logger.info("")
    
    try:
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=str(vae_model_dir),
            local_dir_use_symlinks=False,
        )
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ VAE 모델 다운로드 완료!")
        logger.info("=" * 60)
        logger.info(f"저장 위치: {vae_model_dir.resolve()}")
        logger.info("")
        logger.info("이제 MuseTalk 립싱크가 완전히 작동할 수 있습니다!")
        return True
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 60)
        logger.error("❌ VAE 모델 다운로드 실패")
        logger.error("=" * 60)
        logger.error(f"오류: {e}")
        logger.error("")
        logger.error("수동 다운로드:")
        logger.error("1. https://huggingface.co/stabilityai/sd-vae-ft-mse 방문")
        logger.error("2. 또는 다음 명령 실행:")
        logger.error(f"   python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='stabilityai/sd-vae-ft-mse', local_dir='{vae_model_dir}')\"")
        return False

if __name__ == "__main__":
    success = download_vae_model()
    exit(0 if success else 1)

