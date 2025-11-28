#!/usr/bin/env python3
"""
GPU 및 CUDA 상태 확인 스크립트
"""

import torch

print("=" * 50)
print("PyTorch GPU 상태 확인")
print("=" * 50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
    # 간단한 GPU 테스트
    print("\n" + "=" * 50)
    print("GPU 테스트 실행 중...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = x @ y
        print("✓ GPU 계산 성공!")
    except Exception as e:
        print(f"✗ GPU 계산 실패: {e}")
else:
    print("\n경고: CUDA를 사용할 수 없습니다.")
    print("다음을 확인하세요:")
    print("  1. NVIDIA 드라이버가 설치되어 있는지")
    print("  2. CUDA Toolkit이 설치되어 있는지")
    print("  3. PyTorch CUDA 버전이 설치되어 있는지")

print("=" * 50)

