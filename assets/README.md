# Assets Directory

이 디렉토리는 AI 아바타 서비스에 필요한 에셋을 저장합니다.

## 디렉토리 구조

```
assets/
├── avatars/           # 아바타 소스 이미지
├── idle_loops/        # 감정별 idle 루프 영상
├── voice_samples/     # TTS 음성 클로닝용 샘플
└── README.md
```

## 1. 아바타 이미지 (avatars/)

아바타 소스 이미지를 저장합니다.

### 요구사항
- 형식: PNG, JPG, JPEG
- 권장 크기: 512x512 또는 1024x1024
- 정면 얼굴 사진 권장
- 배경은 단색이면 좋음

### 샘플 이미지 생성
```bash
python tools/create_sample_avatar.py --output assets/avatars/sample.png
```

### 직접 사용
실제 아바타 이미지를 `assets/avatars/` 디렉토리에 복사합니다:
```bash
cp your_avatar.png assets/avatars/
```

## 2. Idle 루프 영상 (idle_loops/)

감정별 idle 애니메이션 루프 영상을 저장합니다.

### 파일 이름 규칙
- `neutral_idle.mp4` - 기본/중립
- `happy_smile.mp4` - 행복/미소
- `sad_idle.mp4` - 슬픔
- `listening_nod.mp4` - 경청 (끄덕임)
- `thinking.mp4` - 생각 중
- `concerned.mp4` - 걱정
- `sympathetic.mp4` - 공감

### 루프 생성
```bash
# 모든 감정 생성
python tools/generate_idle_loops.py --image assets/avatars/sample.png --output assets/idle_loops/

# 특정 감정만 생성
python tools/generate_idle_loops.py --image assets/avatars/sample.png --emotions neutral happy listening
```

### 요구사항
- 형식: MP4 (H.264)
- 프레임레이트: 30 FPS
- 크기: 512x512
- 길이: 3-5초 (루프)

## 3. 음성 샘플 (voice_samples/)

TTS 음성 클로닝을 위한 참조 음성을 저장합니다.

### 요구사항
- 형식: WAV (16-bit PCM)
- 샘플레이트: 24000 Hz
- 길이: 5-30초
- 명확한 발음, 조용한 환경

### 음성 샘플 준비
```bash
# 기존 파일 처리
python tools/prepare_voice_sample.py --input your_voice.wav --output assets/voice_samples/custom.wav

# 마이크 녹음
python tools/prepare_voice_sample.py --record --duration 10 --output assets/voice_samples/recorded.wav

# 녹음 가이드 보기
python tools/prepare_voice_sample.py --show-script
```

## 빠른 시작

1. 샘플 아바타 생성:
```bash
python tools/create_sample_avatar.py
```

2. Idle 루프 생성:
```bash
python tools/generate_idle_loops.py --image assets/avatars/sample_avatar.png
```

3. 서버 시작:
```bash
uvicorn src.main:app --reload
```

## 상업적 사용

모든 생성 도구는 상업적 사용이 가능한 라이브러리만 사용합니다:
- MediaPipe (Apache 2.0)
- OpenCV (Apache 2.0)
- NumPy (BSD)

InsightFace는 비상업적 라이선스이므로 사용하지 않습니다.
