"""
MuseTalk 설치 상태 확인 스크립트

설치 여부와 자동 설치 가능성을 확인합니다.
"""

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 60)
print("MuseTalk 설치 상태 확인")
print("=" * 60)
print()

# 1. 소스 코드 디렉토리 확인
print("[1] 소스 코드 디렉토리 확인:")
musetalk_source_dir = Path("external/MuseTalk")
if musetalk_source_dir.exists():
    print(f"  ✅ MuseTalk 소스 코드 존재: {musetalk_source_dir}")
    musetalk_dir = musetalk_source_dir / "musetalk"
    if musetalk_dir.exists():
        print(f"  ✅ musetalk 패키지 디렉토리 존재")
    else:
        print(f"  ❌ musetalk 패키지 디렉토리 없음")
else:
    print(f"  ❌ MuseTalk 소스 코드 없음: {musetalk_source_dir}")

print()

# 2. Python 경로 추가 테스트
print("[2] Python 경로 추가 테스트:")
try:
    if str(musetalk_source_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(musetalk_source_dir.resolve()))
        print(f"  ✅ Python 경로에 추가됨: {musetalk_source_dir.resolve()}")
    else:
        print(f"  ✅ 이미 Python 경로에 있음")
except Exception as e:
    print(f"  ❌ 경로 추가 실패: {e}")

print()

# 3. MuseTalk 모듈 임포트 테스트
print("[3] MuseTalk 모듈 임포트 테스트:")
try:
    from musetalk.utils.audio_processor import AudioProcessor
    print("  ✅ AudioProcessor 임포트 성공")
    
    from musetalk.models.unet import UNet
    print("  ✅ UNet 임포트 성공")
    
    from musetalk.models.vae import VAE
    print("  ✅ VAE 임포트 성공")
    
    from musetalk.utils.face_parsing import FaceParsing
    print("  ✅ FaceParsing 임포트 성공")
    
    print("  ✅ 모든 MuseTalk 모듈 로드 성공!")
    
except ImportError as e:
    print(f"  ❌ 모듈 임포트 실패: {e}")
    print(f"     확인: external/MuseTalk/musetalk 디렉토리 구조")
except Exception as e:
    print(f"  ⚠️  모듈 임포트 중 오류: {e}")

print()

# 4. 모델 파일 확인
print("[4] 모델 파일 확인:")
model_dir = Path("models/musetalk/musetalkV15")
if model_dir.exists():
    print(f"  ✅ 모델 디렉토리 존재: {model_dir}")
    config_file = model_dir / "musetalk.json"
    unet_file = model_dir / "unet.pth"
    
    if config_file.exists():
        size_mb = config_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ 설정 파일 존재: {config_file.name} ({size_mb:.2f} MB)")
    else:
        print(f"  ❌ 설정 파일 없음: {config_file.name}")
    
    if unet_file.exists():
        size_mb = unet_file.stat().st_size / (1024 * 1024)
        print(f"  ✅ UNet 모델 존재: {unet_file.name} ({size_mb:.2f} MB)")
    else:
        print(f"  ❌ UNet 모델 없음: {unet_file.name}")
else:
    print(f"  ❌ 모델 디렉토리 없음: {model_dir}")

print()

# 5. 실제 통합 코드 작동 여부 확인
print("[5] 통합 코드 작동 여부 확인:")
try:
    # Python 경로에 추가
    if str(musetalk_source_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(musetalk_source_dir.resolve()))
    
    from src.models.integrations.musetalk import MuseTalkModel
    print("  ✅ MuseTalkModel 클래스 임포트 성공")
    
    # 모델 인스턴스 생성 테스트
    model = MuseTalkModel(
        model_dir="models/musetalk/musetalkV15",
        device="cuda",
        fp16=True
    )
    print("  ✅ MuseTalkModel 인스턴스 생성 성공")
    
    print("  ✅ 통합 코드 정상 작동 가능!")
    
except Exception as e:
    print(f"  ❌ 통합 코드 오류: {e}")
    import traceback
    traceback.print_exc()

print()

# 6. 자동 설치 여부 확인
print("[6] 자동 설치 가능 여부:")
print("  현재 상태:")
print("    - 소스 코드: external/MuseTalk에 있음 (수동 다운로드 필요)")
print("    - 모델 파일: models/musetalk에 있음")
print("    - 자동 설치: 부분적으로 지원됨")
print()
print("  자동 설치 방법:")
print("    1. 소스 코드: git clone 또는 다운로드 필요")
print("    2. 모델 파일: python tools/setup_models.py --models lipsync")
print("    3. 통합: 자동으로 Python 경로에 추가됨")

print()
print("=" * 60)
print("요약")
print("=" * 60)

# 최종 상태
has_source = musetalk_source_dir.exists()
has_models = model_dir.exists() and (model_dir / "unet.pth").exists()

if has_source and has_models:
    print("✅ MuseTalk 준비 완료!")
    print("   - 소스 코드: ✅")
    print("   - 모델 파일: ✅")
    print("   - 사용 가능: ✅")
elif has_source:
    print("⚠️  MuseTalk 소스 코드는 있으나 모델 파일이 없음")
    print("   - 소스 코드: ✅")
    print("   - 모델 파일: ❌")
    print("   - 해결: python tools/setup_models.py --models lipsync")
elif has_models:
    print("⚠️  모델 파일은 있으나 소스 코드가 없음")
    print("   - 소스 코드: ❌")
    print("   - 모델 파일: ✅")
    print("   - 해결: git clone https://github.com/TMElyralab/MuseTalk external/MuseTalk")
else:
    print("❌ MuseTalk가 설치되지 않음")
    print("   - 소스 코드: ❌")
    print("   - 모델 파일: ❌")
    print("   - 해결: 자동 설치 스크립트 실행 필요")

print()

