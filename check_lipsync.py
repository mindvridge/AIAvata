"""립싱크 기능 상태 확인"""

print("=" * 60)
print("립싱크 기능 상태 확인")
print("=" * 60)
print()

# 1. MuseTalk 패키지 확인
print("[1] MuseTalk 패키지:")
try:
    import musetalk
    print("  ✅ musetalk 패키지 설치됨")
    print(f"     경로: {musetalk.__file__}")
except ImportError:
    print("  ❌ musetalk 패키지 설치 안 됨")
    print("     → fallback 구현 사용 중")

print()

# 2. 모델 파일 확인
print("[2] MuseTalk 모델 파일:")
import os
from pathlib import Path

model_dir = Path("models/musetalk/musetalkV15")
if model_dir.exists():
    print(f"  ✅ 모델 디렉토리 존재: {model_dir}")
    files = list(model_dir.glob("*"))
    for f in files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"     - {f.name}: {size_mb:.2f} MB")
else:
    print(f"  ❌ 모델 디렉토리 없음: {model_dir}")

print()

# 3. 실제 립싱크 사용 여부 확인
print("[3] 립싱크 사용 여부:")
print("  현재: fallback 구현 사용 (모델이 있어도 패키지 없으면 작동 안 함)")
print("  오디오 입력 → 기본 애니메이션만 적용")
print("  실제 립싱크를 위해서는 MuseTalk 패키지 설치 필요")

print()
print("=" * 60)

