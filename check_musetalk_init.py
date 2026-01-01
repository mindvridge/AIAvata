#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MuseTalk 모델 초기화 상태 확인"""

import sys
from pathlib import Path

print("=" * 60)
print("MuseTalk 모델 초기화 상태 확인")
print("=" * 60)
print()

# 1. 소스 코드 확인
print("[1] 소스 코드 확인:")
source_dir = Path("external/MuseTalk")
if source_dir.exists():
    print(f"  OK: {source_dir}")
    musetalk_pkg = source_dir / "musetalk"
    if musetalk_pkg.exists():
        print(f"  OK: musetalk 패키지 디렉토리 존재")
    else:
        print(f"  ERROR: musetalk 패키지 디렉토리 없음")
else:
    print(f"  ERROR: 소스 코드 없음")

print()

# 2. 모델 파일 확인
print("[2] 모델 파일 확인:")
model_dir = Path("models/musetalk/musetalkV15")
if model_dir.exists():
    print(f"  OK: 모델 디렉토리 존재")
    
    config_file = model_dir / "musetalk.json"
    unet_file = model_dir / "unet.pth"
    
    if config_file.exists():
        size_mb = config_file.stat().st_size / (1024 * 1024)
        print(f"  OK: musetalk.json ({size_mb:.2f} MB)")
    else:
        print(f"  ERROR: musetalk.json 없음")
    
    if unet_file.exists():
        size_mb = unet_file.stat().st_size / (1024 * 1024)
        print(f"  OK: unet.pth ({size_mb:.2f} MB)")
    else:
        print(f"  ERROR: unet.pth 없음")
else:
    print(f"  ERROR: 모델 디렉토리 없음")

print()

# 3. VAE 모델 확인
print("[3] VAE 모델 확인:")
vae_dir = Path("models/musetalk/sd-vae-ft-mse")
if vae_dir.exists():
    print(f"  OK: VAE 디렉토리 존재")
    config = vae_dir / "config.json"
    if config.exists():
        print(f"  OK: config.json 존재")
    else:
        print(f"  WARNING: config.json 없음")
else:
    print(f"  ERROR: VAE 디렉토리 없음")

print()

# 4. Python 경로 추가 테스트
print("[4] Python 경로 추가 테스트:")
try:
    if str(source_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(source_dir.resolve()))
        print(f"  OK: Python 경로에 추가됨")
    else:
        print(f"  OK: 이미 Python 경로에 있음")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# 5. 모듈 임포트 테스트
print("[5] 모듈 임포트 테스트:")
try:
    from musetalk.utils.audio_processor import AudioProcessor
    print("  OK: AudioProcessor")
    
    from musetalk.models.unet import UNet
    print("  OK: UNet")
    
    from musetalk.models.vae import VAE
    print("  OK: VAE")
    
    print("  OK: 모든 모듈 로드 성공!")
    
except ImportError as e:
    print(f"  ERROR: 모듈 임포트 실패: {e}")
except Exception as e:
    print(f"  ERROR: {e}")

print()

# 6. 통합 코드 테스트
print("[6] 통합 코드 테스트:")
try:
    from src.models.integrations.musetalk import MuseTalkModel
    print("  OK: MuseTalkModel 클래스 임포트 성공")
    
    model = MuseTalkModel(
        model_dir="models/musetalk/musetalkV15",
        device="cuda",
        fp16=True
    )
    print("  OK: MuseTalkModel 인스턴스 생성 성공")
    
    # 비동기 초기화 테스트
    import asyncio
    async def test_init():
        result = await model.initialize()
        print(f"  초기화 결과: {result}")
        print(f"  AudioProcessor: {model._audio_processor is not None}")
        print(f"  UNet: {model._unet is not None}")
        print(f"  VAE: {model._vae is not None}")
        return result
    
    result = asyncio.run(test_init())
    if result:
        print("  OK: MuseTalk 모델 초기화 성공!")
    else:
        print("  ERROR: MuseTalk 모델 초기화 실패")
        
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)

