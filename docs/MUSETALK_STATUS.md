# MuseTalk 립싱크 상태 분석

## 📊 현재 상태

### ✅ 작동하는 부분
1. **모델 로드 성공**
   - UNet 모델 (3.2GB) ✅
   - VAE 모델 ✅
   - AudioProcessor 초기화 ✅

2. **초기화 성공**
   - 모든 컴포넌트 정상 초기화 ✅
   - 모델 디렉토리 존재 확인 ✅

### ❌ 문제점

1. **실시간 오디오 처리 미구현**
   - `AudioProcessor.get_audio_feature()`는 파일 경로만 받음
   - 실시간 오디오 스트림 처리 불가능
   - 현재는 fallback 시뮬레이션 사용 중

2. **추론 파이프라인 불완전**
   - 실제 MuseTalk은 Whisper encoder를 사용
   - Face parsing 단계 누락
   - Blending 단계 누락
   - 실제 추론 결과가 제대로 생성되지 않음

3. **성능 이슈**
   - 실시간 30 FPS 목표 달성 어려움
   - 각 프레임마다 전체 추론 파이프라인 실행 필요

## 🔍 기술적 분석

### MuseTalk 실제 파이프라인

```python
# 1. 오디오 특징 추출 (파일 기반)
whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)

# 2. Whisper encoder로 오디오 특징 변환
audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states

# 3. UNet 추론
pred_latents = unet(latent_input, timesteps, encoder_hidden_states=audio_features)

# 4. VAE 디코딩
recon = vae.decode_latents(pred_latents)

# 5. Face parsing & Blending
mask, crop_box = get_image_prepare_material(...)
combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
```

### 현재 구현의 문제

1. **오디오 특징 추출**
   ```python
   # 현재: 간단한 feature_extractor 직접 사용
   audio_feature = self._audio_processor.feature_extractor(audio, ...)
   
   # 필요한 것: Whisper encoder 사용
   audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
   ```

2. **실시간 처리**
   - 현재: 파일 경로 기반 배치 처리
   - 필요: 오디오 스트림 기반 실시간 처리

3. **Face Parsing & Blending**
   - 현재: 누락됨
   - 필요: MuseTalk의 blending 파이프라인 통합

## 🛠️ 해결 방안

### 옵션 1: 실시간 오디오 처리 개선 (권장)

```python
# feature_extractor를 직접 사용하여 실시간 청크 처리
def _extract_audio_features_realtime(self, audio_chunk, sample_rate):
    # 1. 리샘플링 (16kHz)
    audio = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=16000)
    
    # 2. Feature extractor 사용
    audio_feature = self._audio_processor.feature_extractor(
        audio,
        return_tensors="pt",
        sampling_rate=16000
    ).input_features
    
    # 3. Whisper encoder (필요시)
    # audio_feats = whisper.encoder(audio_feature, ...)
    
    return audio_feature
```

### 옵션 2: MuseTalk realtime_inference 통합

- `external/MuseTalk/scripts/realtime_inference.py` 참고
- 실시간 오디오 큐 처리 방식 사용
- 전체 파이프라인 재구현

### 옵션 3: 현재 상태 유지 (Fallback 사용)

- 현재 시뮬레이션 방식 유지
- 실제 립싱크 품질은 낮지만 동작은 함

## 📝 권장 사항

1. **단기**: 실시간 오디오 특징 추출 개선
   - `feature_extractor` 직접 사용
   - 오디오 청크 단위 처리

2. **중기**: MuseTalk 파이프라인 완전 통합
   - Whisper encoder 추가
   - Face parsing & Blending 추가

3. **장기**: 성능 최적화
   - 배치 처리
   - 캐싱
   - GPU 최적화

## 🔗 참고 자료

- MuseTalk GitHub: https://github.com/TMElyralab/MuseTalk
- Realtime Inference: `external/MuseTalk/scripts/realtime_inference.py`
- Audio Processor: `external/MuseTalk/musetalk/utils/audio_processor.py`

