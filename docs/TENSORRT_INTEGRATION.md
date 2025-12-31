# TensorRT Integration for MuseTalk

MuseTalk에 NVIDIA TensorRT를 적용하여 추론 속도를 크게 향상시킬 수 있습니다.

## 개요

TensorRT는 NVIDIA의 고성능 딥러닝 추론 엔진으로, 모델을 최적화하여 추론 속도를 최대 36배까지 향상시킬 수 있습니다.

## 설치

### 1. TensorRT 설치

TensorRT를 사용하려면 다음 중 하나를 설치해야 합니다:

**옵션 1: torch-tensorrt (권장)**
```bash
pip install torch-tensorrt
```

**옵션 2: NVIDIA TensorRT Python API**
```bash
# CUDA, cuDNN이 설치되어 있어야 함
# TensorRT tar 파일 다운로드 및 설치 필요
# 자세한 내용: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
```

### 2. 의존성 확인

- CUDA 11.8 이상
- cuDNN 8.6 이상
- PyTorch (CUDA 지원 버전)

## 사용 방법

### 환경 변수 설정

`.env` 파일에 다음 설정 추가:

```env
# TensorRT 설정
MUSETALK_USE_TENSORRT=true
MUSETALK_TENSORRT_WORKSPACE_SIZE=1073741824  # 1GB (바이트)
MUSETALK_TENSORRT_FP16=true  # FP16 정밀도 사용 (더 빠름)
```

또는 `config.py`에서 기본값 변경:

```python
musetalk_use_tensorrt: bool = True
musetalk_tensorrt_workspace_size: int = 1024 * 1024 * 1024  # 1GB
musetalk_tensorrt_fp16: bool = True
```

## 작동 방식

1. **초기화 시 변환**: MuseTalk 모델 초기화 시 UNet 모델이 TensorRT로 변환됩니다.
2. **엔진 캐싱**: 변환된 엔진은 `models/musetalk/tensorrt_engines/` 디렉토리에 저장됩니다.
3. **자동 폴백**: TensorRT 변환이 실패하면 자동으로 PyTorch 추론으로 폴백합니다.

## 성능 향상

TensorRT를 사용하면 다음과 같은 성능 향상을 기대할 수 있습니다:

- **UNet 추론 속도**: 2-5배 향상 (FP16 사용 시)
- **전체 립싱크 파이프라인**: 1.5-3배 향상
- **GPU 메모리 사용량**: 약간 증가 (엔진 최적화 과정에서)

## 주의사항

1. **첫 실행 지연**: 첫 번째 실행 시 TensorRT 엔진 컴파일에 수 분이 소요될 수 있습니다.
2. **입력 크기 고정**: TensorRT 엔진은 특정 입력 크기에 최적화됩니다. 다른 크기의 입력이 필요하면 새 엔진이 생성됩니다.
3. **메모리 요구사항**: TensorRT 변환 과정에서 추가 GPU 메모리가 필요합니다 (최소 2GB 권장).

## 트러블슈팅

### TensorRT 변환 실패

로그에서 다음 메시지 확인:
```
⚠️ TensorRT conversion failed: ...
Falling back to PyTorch inference
```

**해결 방법**:
- `torch-tensorrt`가 올바르게 설치되었는지 확인
- CUDA 및 cuDNN 버전 확인
- GPU 메모리 충분한지 확인

### 엔진 캐싱 문제

엔진 파일 손상 시 `models/musetalk/tensorrt_engines/` 디렉토리 삭제 후 재생성:

```bash
rm -rf models/musetalk/tensorrt_engines/
```

### 성능 향상이 미미한 경우

- FP16 정밀도 사용 확인 (`MUSETALK_TENSORRT_FP16=true`)
- 배치 크기 확인 (현재는 배치 크기 1만 지원)
- GPU 모델 확인 (최신 GPU에서 더 큰 향상)

## 향후 개선 사항

- [ ] VAE 디코더 TensorRT 변환 지원
- [ ] 동적 배치 크기 지원
- [ ] INT8 정량화 지원 (더 큰 속도 향상)
- [ ] 엔진 재사용 최적화
