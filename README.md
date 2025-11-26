# Realtime AI Avatar Service

HeyGen과 유사한 실시간 대화형 AI 아바타 서비스입니다. 사용자가 음성으로 대화하면 AI 아바타가 자연스러운 립싱크와 표정으로 실시간 응답합니다.

## 핵심 목표

- 첫 응답 지연시간: 800ms 이내
- 프레임 레이트: 30 FPS
- 모든 컴포넌트 상업적 사용 가능 (MIT/Apache 2.0)
- 한국어 우선 지원

## 기술 스택

| 구분 | 기술 | 용도 | 라이선스 |
|------|------|------|----------|
| STT | SenseVoice-Small | 음성→텍스트 + 감정인식 | Apache 2.0 |
| LLM | Claude API | 대화 응답 생성 | 상용 API |
| TTS | Chatterbox | 텍스트→음성 (스트리밍) | MIT |
| 립싱크 | MuseTalk 1.5 | 실시간 립싱크 오버레이 | MIT |
| Idle 생성 | LivePortrait + MediaPipe | 사전 생성 idle 루프 | MIT + Apache 2.0 |
| 얼굴 감지 | MediaPipe | 얼굴 랜드마크 감지 | Apache 2.0 |
| VAD | Silero VAD | 음성 활동 감지 | MIT |
| 스트리밍 | LiveKit | WebRTC 서버 | Apache 2.0 |

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                        사용자 브라우저                                │
│                    (WebRTC 양방향 스트리밍)                           │
└─────────────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────────────┐
│                         LiveKit Server                              │
│                      (WebRTC 미디어 서버)                            │
└─────────────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend Server                          │
│  ┌──────────┐   ┌─────────┐   ┌───────────┐   ┌─────────────────┐  │
│  │   STT    │ → │   LLM   │ → │    TTS    │ → │  Avatar Render  │  │
│  │SenseVoice│   │ Claude  │   │Chatterbox │   │MuseTalk + Idle  │  │
│  │  Small   │   │   API   │   │           │   │     Loops       │  │
│  └──────────┘   └─────────┘   └───────────┘   └─────────────────┘  │
│       ↓                                              ↑              │
│   [감정 정보] ──────────────────────────→ [Idle 루프 선택]          │
└─────────────────────────────────────────────────────────────────────┘
```

## 빠른 시작

### 1. 환경 설정

```bash
# Python 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 입력
```

### 2. 모델 다운로드

```bash
python tools/setup_models.py
```

### 3. Idle 루프 생성

```bash
python tools/generate_idle_loops.py --image assets/avatars/avatar.jpg --output assets/idle_loops/
```

### 4. 서버 실행

```bash
# 개발 모드
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# 프로덕션 (Docker)
docker-compose up -d
```

### 5. 프론트엔드 실행

```bash
cd frontend
npm install
npm run dev
```

## API 엔드포인트

### REST API

- `GET /health` - 서버 상태 확인
- `POST /api/generate-token` - LiveKit 토큰 생성
- `POST /api/avatar/create` - 새 아바타 세션 생성
- `GET /api/avatar/{session_id}` - 세션 정보 조회

### WebSocket

- `WS /ws/avatar` - 실시간 아바타 통신
  - Client → Server: 오디오 청크 (bytes)
  - Server → Client: 비디오 프레임 (bytes)

## 프로젝트 구조

```
realtime-avatar/
├── src/
│   ├── main.py                    # FastAPI 앱 엔트리포인트
│   ├── config.py                  # 설정 관리
│   ├── pipeline/
│   │   ├── orchestrator.py        # 전체 파이프라인 조율
│   │   ├── stt_module.py          # SenseVoice STT + 감정인식
│   │   ├── llm_module.py          # LLM 스트리밍 응답
│   │   ├── tts_module.py          # Chatterbox TTS 스트리밍
│   │   └── avatar_renderer.py     # MuseTalk 립싱크 + Idle 루프
│   ├── models/
│   │   ├── schemas.py             # Pydantic 모델
│   │   └── emotion.py             # 감정 enum 및 매핑
│   ├── services/
│   │   ├── livekit_service.py     # WebRTC 연결 관리
│   │   ├── tts_cache_service.py   # TTS 캐싱
│   │   └── idle_loop_manager.py   # Idle 루프 관리
│   ├── utils/
│   │   ├── audio_utils.py         # 오디오 처리 유틸
│   │   ├── video_utils.py         # 비디오 처리 유틸
│   │   └── vad.py                 # Silero VAD 래퍼
│   └── api/
│       ├── routes.py              # API 엔드포인트
│       └── websocket.py           # WebSocket 핸들러
├── tools/
│   ├── generate_idle_loops.py     # Idle 루프 생성 스크립트
│   ├── setup_models.py            # 모델 다운로드 스크립트
│   └── benchmark.py               # 성능 벤치마크
├── assets/
│   ├── avatars/                   # 아바타 원본 이미지
│   └── idle_loops/                # 생성된 idle 루프 영상
├── tests/
├── frontend/                      # React 프론트엔드
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 사용된 모든 컴포넌트는 상업적 사용이 가능합니다.

### 사용 금지 컴포넌트

- **InsightFace 모델**: 비상업적 전용 라이선스
- **JoyVASA**: 라이선스 미명시 + InsightFace 의존

## 참고 링크

- [SenseVoice GitHub](https://github.com/FunAudioLLM/SenseVoice)
- [MuseTalk GitHub](https://github.com/TMElyralab/MuseTalk)
- [LivePortrait GitHub](https://github.com/KwaiVGI/LivePortrait)
- [Chatterbox TTS](https://github.com/resemble-ai/chatterbox)
- [LiveKit Docs](https://docs.livekit.io/)
- [MediaPipe Face Mesh](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
