# VAE (Variational Autoencoder)란?

## MuseTalk에서 VAE의 역할

MuseTalk 립싱크 파이프라인은 다음과 같이 작동합니다:

```
1. 얼굴 이미지 → VAE 인코딩 → Latent (압축된 표현)
2. Latent + 오디오 특징 → UNet 추론 → 새로운 Latent (립싱크 적용)
3. 새로운 Latent → VAE 디코딩 → 립싱크된 이미지
```

## 왜 필요한가?

- **이미지 압축**: 고해상도 이미지를 작은 latent space로 압축하여 UNet이 빠르게 처리
- **고품질 디코딩**: UNet의 출력을 다시 고품질 이미지로 복원
- **필수 컴포넌트**: MuseTalk의 실제 추론 파이프라인에 필수

## 현재 상태

- ❌ VAE 모델이 설치되지 않음
- ✅ UNet은 있지만 VAE 없이는 완전한 추론 불가
- ⚠️ 현재는 시뮬레이션만 작동

## 해결 방법

VAE 모델을 다운로드해야 합니다:
- 모델: `sd-vae-ft-mse` (Stable Diffusion VAE)
- 크기: 약 335MB
- 저장 위치: `models/sd-vae-ft-mse/`

