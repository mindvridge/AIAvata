"""MuseTalk 실제 사용 여부 확인 테스트"""

import sys
from pathlib import Path

# MuseTalk 경로 추가
sys.path.insert(0, str(Path("external/MuseTalk").resolve()))

print("=" * 60)
print("MuseTalk 실제 사용 여부 확인")
print("=" * 60)

# 1. 모델 로드 확인
print("\n[1] MuseTalk 모델 초기화 확인...")
try:
    from src.models.integrations.musetalk import MuseTalkModel
    
    model = MuseTalkModel()
    print(f"✅ MuseTalkModel 생성 성공")
    print(f"   - AudioProcessor 속성: {hasattr(model, '_audio_processor')}")
    print(f"   - UNet 속성: {hasattr(model, '_unet')}")
    print(f"   - process_frame 메서드: {hasattr(model, 'process_frame')}")
    
    # 초기화 확인
    import asyncio
    async def check_init():
        result = await model.initialize()
        print(f"\n[2] 초기화 결과: {result}")
        print(f"   - AudioProcessor: {model._audio_processor is not None}")
        print(f"   - UNet: {model._unet is not None}")
        print(f"   - VAE: {model._vae is not None}")
        print(f"   - 초기화 완료: {model._initialized}")
        
        # process_frame 메서드 확인
        print(f"\n[3] process_frame 메서드 확인...")
        import inspect
        source = inspect.getsource(model.process_frame)
        print(f"   - 메서드 존재: ✅")
        
        # AudioProcessor 메서드 확인
        if model._audio_processor:
            print(f"\n[4] AudioProcessor 메서드 확인...")
            ap_methods = [m for m in dir(model._audio_processor) if not m.startswith('_')]
            print(f"   - 사용 가능한 메서드: {ap_methods}")
            
            # get_audio_feature는 파일 경로를 받음
            if 'get_audio_feature' in ap_methods:
                print(f"   ⚠️  get_audio_feature는 파일 경로를 받습니다 (실시간 처리 어려움)")
    
    asyncio.run(check_init())
    
except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("결론:")
print("=" * 60)
print("현재 MuseTalk은 초기화되지만, 실제 추론은 구현이 불완전합니다.")
print("process_frame 메서드가 실제 MuseTalk 파이프라인과 맞지 않습니다.")
print("\n해결 방법:")
print("1. AudioProcessor는 파일 경로를 요구하므로 임시 파일 방식 사용")
print("2. 또는 실제 MuseTalk realtime_inference 코드를 참고하여 재구현")
print("3. 현재는 시뮬레이션으로 fallback됨")

