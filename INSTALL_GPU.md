# GPU 지원 설치 가이드

이 가이드는 NVIDIA GPU를 사용하기 위한 CUDA 드라이버와 PyTorch CUDA 버전 설치 방법을 설명합니다.

## 현재 시스템 상태

- **GPU**: NVIDIA GeForce RTX 5060 Ti
- **CUDA 드라이버 버전**: 581.57 (CUDA 13.0 지원)
- **현재 PyTorch**: CPU 버전 (2.7.1+cpu)

## 설치 방법

### 방법 1: PowerShell 스크립트 사용 (권장)

```powershell
.\install_pytorch_gpu.ps1
```

### 방법 2: 수동 설치

1. **기존 CPU 버전 제거**
   ```powershell
   pip uninstall torch torchaudio -y
   ```

2. **PyTorch CUDA 12.1 설치**
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   또는 CUDA 12.4 버전 (더 최신):
   ```powershell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

### 방법 3: requirements-gpu.txt 사용

```powershell
pip install -r requirements-gpu.txt --index-url https://download.pytorch.org/whl/cu121
```

## 설치 확인

설치 후 다음 명령어로 확인하세요:

```python
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
```

## CUDA 버전 호환성

- **시스템 CUDA**: 13.0 (드라이버 581.57)
- **PyTorch CUDA**: 12.1 또는 12.4 (권장)
- **호환성**: CUDA 13.0은 CUDA 12.1/12.4와 하위 호환되므로 정상 작동합니다.

## 문제 해결

### 1. CUDA가 감지되지 않는 경우

```powershell
# NVIDIA 드라이버 재확인
nvidia-smi

# CUDA Toolkit 확인
nvcc --version
```

### 2. PyTorch가 GPU를 찾지 못하는 경우

- NVIDIA 드라이버를 최신 버전으로 업데이트
- Python 환경을 다시 시작
- PyTorch를 재설치

### 3. 다른 CUDA 버전이 필요한 경우

PyTorch 공식 웹사이트에서 호환 버전 확인:
https://pytorch.org/get-started/locally/

## 참고 사항

- CPU 버전과 CUDA 버전은 동시에 설치할 수 없습니다.
- Docker를 사용하는 경우, `docker-compose.yml`에서 GPU 지원을 활성화해야 합니다.
- 프로덕션 환경에서는 `requirements-gpu.txt`를 사용하세요.

