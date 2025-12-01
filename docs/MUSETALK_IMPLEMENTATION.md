# MuseTalk 립싱크 구현 완료

## 📋 구현 내용

### ✅ 완료된 기능

1. **Whisper Encoder 통합**
   - Whisper 모델 자동 로드
   - 실시간 오디오 특징 추출
   - Hidden states 추출 및 처리

2. **실시간 오디오 처리**
   - 오디오 버퍼링 (2초 버퍼)
   - 16kHz 리샘플링
   - 프레임별 오디오 청크 처리

3. **UNet 추론 파이프라인**
   - 올바른 UNet 호출 방식
   - PositionalEncoding 적용
   - 디바이스 및 dtype 관리

4. **VAE 디코딩**
   - 얼굴 이미지 latent 인코딩
   - 디코딩 및 후처리
   - 원본 크기로 리사이즈

## 🔧 주요 변경사항

### 1. Whisper 모델 추가
```python
# Whisper encoder 초기화
self._whisper = WhisperModel.from_pretrained(whisper_path)
self._whisper = self._whisper.to(device, dtype=weight_dtype).eval()
```

### 2. 실시간 오디오 특징 추출
```python
# Whisper encoder로 hidden states 추출
audio_feats = self._whisper.encoder(audio_feature, output_hidden_states=True).hidden_states
audio_feats = torch.stack(audio_feats, dim=2)  # 모든 레이어 스택
```

### 3. 오디오 버퍼링
```python
# 실시간 처리를 위한 오디오 버퍼
self._audio_buffer.append(audio.copy())
audio_combined = np.concatenate(self._audio_buffer)
```

### 4. 올바른 UNet 추론
```python
# PositionalEncoding 적용
audio_features = self._positional_encoding(audio_features)

# UNet 추론
pred_latents = self._unet.model(
    latent_input,
    timesteps,
    encoder_hidden_states=audio_features
).sample
```

## 📝 사용 방법

### 기본 사용
```python
from src.models.integrations.musetalk import MuseTalkModel

# 모델 초기화
model = MuseTalkModel(
    model_dir="models/musetalk/musetalkV15",
    device="cuda",
    fp16=True
)
await model.initialize()

# 립싱크 처리
result_frame = await model.process_frame(
    source_frame=face_image,  # numpy array (BGR)
    audio_chunk=audio_data,   # numpy array (float32)
    audio_sample_rate=16000
)
```

### 테스트
```bash
python test_musetalk_realtime.py
```

## ⚠️ 주의사항

1. **Whisper 모델 다운로드**
   - 첫 실행 시 HuggingFace에서 Whisper 모델 자동 다운로드
   - 또는 `models/whisper` 디렉토리에 모델 저장

2. **메모리 사용**
   - GPU 메모리 약 4-6GB 필요 (UNet + VAE + Whisper)
   - 오디오 버퍼는 최대 2초까지 유지

3. **실시간 처리 성능**
   - 각 프레임 처리 시간: 약 50-200ms (GPU 기준)
   - 30 FPS 목표 달성을 위해서는 배치 처리 필요

## 🔄 다음 단계 (선택사항)

1. **Face Parsing & Blending**
   - MuseTalk의 고급 blending 기능 통합
   - 더 자연스러운 입 움직임

2. **배치 처리 최적화**
   - 여러 프레임 동시 처리
   - 성능 향상

3. **캐싱 최적화**
   - 얼굴 latent 캐싱
   - 오디오 특징 캐싱

## 📚 참고 자료

- MuseTalk GitHub: https://github.com/TMElyralab/MuseTalk
- 구현 파일: `src/models/integrations/musetalk.py`
- 테스트 스크립트: `test_musetalk_realtime.py`

