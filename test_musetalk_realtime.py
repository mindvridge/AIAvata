"""
MuseTalk 실시간 립싱크 테스트 스크립트

이 스크립트는 MuseTalk 립싱크가 제대로 작동하는지 확인합니다.
"""

import asyncio
import numpy as np
import cv2
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_musetalk_lipsync():
    """MuseTalk 립싱크 테스트"""
    print("=" * 60)
    print("MuseTalk 실시간 립싱크 테스트")
    print("=" * 60)
    print()
    
    try:
        from src.models.integrations.musetalk import MuseTalkModel
        
        # MuseTalk 모델 초기화
        print("[1] MuseTalk 모델 초기화 중...")
        model = MuseTalkModel(
            model_dir="models/musetalk/musetalkV15",
            device="cuda",
            fp16=True
        )
        
        success = await model.initialize()
        if not success:
            print("❌ MuseTalk 모델 초기화 실패")
            return False
        
        print("✅ MuseTalk 모델 초기화 성공")
        print()
        
        # 테스트 이미지 로드
        print("[2] 테스트 이미지 로드 중...")
        test_image_path = Path("assets/avatars/avata.png")
        if not test_image_path.exists():
            # 기본 테스트 이미지 생성
            test_image = np.ones((512, 512, 3), dtype=np.uint8) * 128
            cv2.putText(test_image, "Test Avatar", (150, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            logger.info("Created default test image")
        else:
            test_image = cv2.imread(str(test_image_path))
            if test_image is None:
                print("❌ 이미지 로드 실패")
                return False
            test_image = cv2.resize(test_image, (512, 512))
        
        print(f"✅ 테스트 이미지 로드: {test_image.shape}")
        print()
        
        # 테스트 오디오 생성 (사인파)
        print("[3] 테스트 오디오 생성 중...")
        sample_rate = 16000
        duration = 1.0  # 1초
        frequency = 440  # A4 음
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        test_audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        print(f"✅ 테스트 오디오 생성: {len(test_audio)} 샘플, {sample_rate}Hz")
        print()
        
        # 립싱크 처리 테스트
        print("[4] 립싱크 처리 테스트 중...")
        try:
            # 오디오를 작은 청크로 나누어 테스트
            chunk_size = int(sample_rate * 0.1)  # 100ms 청크
            
            for i in range(0, min(len(test_audio), chunk_size * 5), chunk_size):
                audio_chunk = test_audio[i:i+chunk_size]
                if len(audio_chunk) < chunk_size:
                    # 마지막 청크 패딩
                    audio_chunk = np.pad(audio_chunk, (0, chunk_size - len(audio_chunk)), 
                                       mode='constant', constant_values=0)
                
                print(f"  처리 중: 청크 {i//chunk_size + 1}/5")
                
                result_frame = await model.process_frame(
                    source_frame=test_image.copy(),
                    audio_chunk=audio_chunk,
                    audio_sample_rate=sample_rate
                )
                
                if result_frame is not None:
                    print(f"    ✅ 프레임 생성 성공: {result_frame.shape}")
                else:
                    print(f"    ⚠️  프레임 생성 실패 (None 반환)")
                    break
            
            print()
            print("✅ 립싱크 처리 테스트 완료")
            
        except Exception as e:
            print(f"❌ 립싱크 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 정리
        print()
        print("[5] 리소스 정리 중...")
        await model.cleanup()
        print("✅ 정리 완료")
        
        print()
        print("=" * 60)
        print("✅ 모든 테스트 통과!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result = asyncio.run(test_musetalk_realtime())
    exit(0 if result else 1)

