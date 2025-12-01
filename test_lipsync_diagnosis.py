"""
립싱크 작동 진단 스크립트
"""

import asyncio
import logging
import numpy as np
import cv2
from pathlib import Path

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_lipsync():
    """립싱크 전체 파이프라인 테스트"""
    
    logger.info("=" * 60)
    logger.info("립싱크 진단 테스트 시작")
    logger.info("=" * 60)
    
    # 1. AvatarRenderer 초기화 테스트
    logger.info("\n[1단계] AvatarRenderer 초기화 중...")
    try:
        from src.pipeline.avatar_renderer import AvatarRenderer
        
        renderer = AvatarRenderer()
        await renderer.initialize()
        
        logger.info(f"✅ AvatarRenderer 초기화 성공")
        logger.info(f"   - MuseTalk 모델: {renderer._musetalk_model is not None}")
        
        if renderer._musetalk_model:
            logger.info(f"   - MuseTalk 초기화 상태: {renderer._musetalk_model._initialized}")
        else:
            logger.warning("   ⚠️ MuseTalk 모델이 없습니다!")
            
    except Exception as e:
        logger.error(f"❌ AvatarRenderer 초기화 실패: {e}", exc_info=True)
        return
    
    # 2. 립싱크 적용 테스트
    logger.info("\n[2단계] 립싱크 적용 테스트 중...")
    try:
        # 테스트 프레임 생성
        test_frame = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        # 테스트 오디오 생성 (1 프레임 분량, 30fps 기준)
        samples_per_frame = int(24000 / 30)  # 800 samples
        test_audio = np.random.randn(samples_per_frame).astype(np.float32) * 0.1
        test_audio_bytes = (test_audio * 32767).astype(np.int16).tobytes()
        
        logger.info(f"   테스트 프레임: {test_frame.shape}")
        logger.info(f"   테스트 오디오: {len(test_audio)} samples ({len(test_audio_bytes)} bytes)")
        
        # 립싱크 적용
        result_frame = await renderer._apply_lipsync(test_frame, test_audio_bytes)
        
        logger.info(f"✅ 립싱크 적용 성공")
        logger.info(f"   결과 프레임: {result_frame.shape}")
        
        # 프레임 비교
        if np.array_equal(test_frame, result_frame):
            logger.warning("   ⚠️ 결과 프레임이 원본과 동일합니다 (변화 없음)")
        else:
            diff = np.abs(test_frame.astype(np.float32) - result_frame.astype(np.float32))
            logger.info(f"   평균 차이: {diff.mean():.2f} (최대: {diff.max():.2f})")
            
    except Exception as e:
        logger.error(f"❌ 립싱크 적용 실패: {e}", exc_info=True)
        return
    
    # 3. 오디오 스트림 렌더링 테스트
    logger.info("\n[3단계] 오디오 스트림 렌더링 테스트 중...")
    try:
        async def test_audio_stream():
            """테스트 오디오 스트림 생성"""
            # 3프레임 분량의 오디오 생성
            for i in range(3):
                audio = np.random.randn(samples_per_frame).astype(np.float32) * 0.1
                audio_bytes = (audio * 32767).astype(np.int16).tobytes()
                yield audio_bytes
                await asyncio.sleep(0.01)
        
        frame_count = 0
        async for frame in renderer.render_with_audio(
            audio_stream=test_audio_stream(),
            audio_sample_rate=24000
        ):
            frame_count += 1
            if frame_count >= 3:
                break
        
        logger.info(f"✅ 오디오 스트림 렌더링 성공: {frame_count} 프레임 생성")
        
    except Exception as e:
        logger.error(f"❌ 오디오 스트림 렌더링 실패: {e}", exc_info=True)
        return
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ 모든 테스트 통과!")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_lipsync())

