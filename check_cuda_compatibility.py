"""CUDA 호환성 확인 스크립트"""

import torch
import subprocess
import sys

print("=" * 60)
print("CUDA 호환성 확인")
print("=" * 60)
print()

# 1. PyTorch 정보
print("[1] PyTorch 정보:")
print(f"  PyTorch 버전: {torch.__version__}")
print(f"  CUDA 사용 가능: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  CUDA 버전: {torch.version.cuda}")
    print(f"  cuDNN 버전: {torch.backends.cudnn.version()}")
    print(f"  GPU 개수: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}:")
        print(f"    이름: {torch.cuda.get_device_name(i)}")
        print(f"    Compute Capability: {torch.cuda.get_device_capability(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    메모리: {props.total_memory / 1024**3:.2f} GB")
else:
    print("  CUDA를 사용할 수 없습니다.")
    print()

# 2. NVIDIA 드라이버 정보
print("\n[2] NVIDIA 드라이버 정보:")
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=name,driver_version,compute_cap', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        check=True
    )
    print(result.stdout.strip())
except FileNotFoundError:
    print("  nvidia-smi를 찾을 수 없습니다.")
except subprocess.CalledProcessError as e:
    print(f"  오류: {e}")
except Exception as e:
    print(f"  오류: {e}")

# 3. 호환성 체크
print("\n[3] 호환성 체크:")
if torch.cuda.is_available():
    try:
        # 간단한 텐서 연산 테스트
        x = torch.randn(10, 10).cuda()
        y = torch.randn(10, 10).cuda()
        z = torch.matmul(x, y)
        print("  ✅ CUDA 텐서 연산 테스트 성공")
    except Exception as e:
        print(f"  ❌ CUDA 텐서 연산 테스트 실패: {e}")
        print("  → CUDA 호환성 문제가 있을 수 있습니다.")
else:
    print("  ⚠️  CUDA를 사용할 수 없습니다.")

print()
print("=" * 60)

