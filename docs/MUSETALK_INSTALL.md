# MuseTalk 설치 가이드

## ✅ 현재 상태

**MuseTalk이 제대로 설치되어 있습니다!**

- ✅ 소스 코드: `external/MuseTalk` 디렉토리에 있음
- ✅ 모델 파일: `models/musetalk/musetalkV15` 디렉토리에 있음 (3.2GB)
- ✅ Python 경로: 자동으로 추가됨
- ✅ 통합 코드: 정상 작동

## 📦 자동 설치 여부

### 현재 구현 (부분 자동)

1. **Python 경로 자동 추가**: ✅
   - `src/models/integrations/musetalk.py`에서 자동으로 경로 추가
   - `external/MuseTalk` 디렉토리가 Python 경로에 자동 추가됨

2. **모델 파일 자동 다운로드**: ✅
   - `python tools/setup_models.py --models lipsync` 실행 시
   - HuggingFace에서 모델 자동 다운로드

3. **소스 코드 자동 다운로드**: ✅ (자동 지원)
   - `python tools/setup_models.py --models lipsync` 실행 시
   - Git clone 또는 ZIP 파일 다운로드로 자동 설치
   - 이미 있으면 건너뛰기

## 🔧 설치 확인 방법

```bash
# 설치 상태 확인
python check_musetalk_install.py
```

## 📥 수동 설치 방법

### 1. 소스 코드 다운로드 (자동)

```bash
# 자동 다운로드 (권장)
python tools/setup_models.py --models lipsync

# 또는 수동 다운로드
git clone https://github.com/TMElyralab/MuseTalk external/MuseTalk
```

### 2. 모델 파일 다운로드 (자동 가능)

```bash
# 자동 다운로드 (HuggingFace)
python tools/setup_models.py --models lipsync

# 또는 수동으로 모델 파일 다운로드
# models/musetalk/musetalkV15/ 디렉토리에 배치
```

## 🚀 완전 자동 설치 구현 (개선 가능)

현재는 소스 코드를 수동으로 다운로드해야 하지만, 다음과 같이 자동화할 수 있습니다:

```python
# tools/setup_models.py에 추가 가능
def setup_musetalk_source():
    """MuseTalk 소스 코드 자동 다운로드"""
    import subprocess
    from pathlib import Path
    
    musetalk_dir = Path("external/MuseTalk")
    if musetalk_dir.exists():
        logger.info("MuseTalk source code already exists")
        return True
    
    try:
        logger.info("Cloning MuseTalk repository...")
        subprocess.run([
            "git", "clone",
            "https://github.com/TMElyralab/MuseTalk.git",
            str(musetalk_dir)
        ], check=True)
        logger.info("MuseTalk source code downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to clone MuseTalk: {e}")
        return False
```

## ✅ 설치 확인 체크리스트

- [x] 소스 코드: `external/MuseTalk` 존재
- [x] 모델 파일: `models/musetalk/musetalkV15/unet.pth` 존재 (3.2GB)
- [x] Python 경로: 자동 추가됨
- [x] 모듈 임포트: 정상 작동
- [x] 통합 코드: 정상 작동

## 🔍 문제 해결

### MuseTalk 모듈을 찾을 수 없는 경우

```python
# src/models/integrations/musetalk.py가 자동으로 처리
# 하지만 수동으로 확인하려면:
import sys
from pathlib import Path
sys.path.insert(0, str(Path("external/MuseTalk").resolve()))
from musetalk.utils.audio_processor import AudioProcessor
```

### 모델 파일이 없는 경우

```bash
python tools/setup_models.py --models lipsync
```

### 소스 코드가 없는 경우

```bash
git clone https://github.com/TMElyralab/MuseTalk external/MuseTalk
```

## 📚 참고 자료

- MuseTalk GitHub: https://github.com/TMElyralab/MuseTalk
- 설치 확인 스크립트: `check_musetalk_install.py`
- 모델 설정 스크립트: `tools/setup_models.py`

