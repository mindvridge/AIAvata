"""
MuseTalk 소스 코드 자동 다운로드 테스트

이 스크립트는 소스 코드 자동 다운로드 기능이 작동하는지 확인합니다.
"""

import sys
from pathlib import Path

# tools 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.setup_models import download_musetalk_source
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 60)
print("MuseTalk 소스 코드 자동 다운로드 테스트")
print("=" * 60)
print()

# 테스트 디렉토리 (기존 디렉토리는 보존)
test_dir = Path("external/MuseTalk_test")

print(f"[테스트] 다운로드 대상: {test_dir}")
print()

# 기존 테스트 디렉토리가 있으면 제거
if test_dir.exists():
    print(f"기존 테스트 디렉토리 발견: {test_dir}")
    response = input("제거하고 다시 다운로드하시겠습니까? (y/N): ")
    if response.lower() == 'y':
        import shutil
        shutil.rmtree(test_dir)
        print("제거 완료")
    else:
        print("기존 디렉토리 유지. 테스트 종료.")
        sys.exit(0)

print()
print("다운로드 시작...")
print()

# 다운로드 실행
success = download_musetalk_source(test_dir)

print()
print("=" * 60)
if success:
    print("✅ 다운로드 성공!")
    print()
    print("다운로드된 파일 확인:")
    if test_dir.exists():
        musetalk_dir = test_dir / "musetalk"
        if musetalk_dir.exists():
            print(f"  ✅ musetalk 패키지 디렉토리 존재")
            files = list(musetalk_dir.glob("**/*.py"))[:5]
            for f in files:
                print(f"     - {f.relative_to(test_dir)}")
        else:
            print(f"  ❌ musetalk 패키지 디렉토리 없음")
        
        print()
        print(f"테스트 디렉토리: {test_dir}")
        print("실제 사용 시에는 external/MuseTalk로 다운로드됩니다.")
    else:
        print("  ⚠️  디렉토리가 생성되지 않았습니다.")
else:
    print("❌ 다운로드 실패")
    print()
    print("수동 다운로드 방법:")
    print("  git clone https://github.com/TMElyralab/MuseTalk external/MuseTalk")

print("=" * 60)

