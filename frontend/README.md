# Realtime AI Avatar Frontend

React + TypeScript 기반 실시간 AI 아바타 프론트엔드입니다.

## 기술 스택

- **React 18** - UI 프레임워크
- **TypeScript** - 타입 안전성
- **Vite** - 빌드 도구
- **TailwindCSS** - 스타일링
- **LiveKit Client** - WebRTC 스트리밍
- **Lucide React** - 아이콘

## 설치 및 실행

### 개발 환경

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

개발 서버가 http://localhost:3000 에서 실행됩니다.

### 프로덕션 빌드

```bash
# 빌드
npm run build

# 미리보기
npm run preview
```

## 환경 변수

`.env` 파일을 생성하고 설정:

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_LIVEKIT_URL=ws://localhost:7880
```

## 프로젝트 구조

```
src/
├── components/       # React 컴포넌트
│   ├── AvatarView.tsx     # 아바타 비디오 표시
│   ├── AudioRecorder.tsx  # 마이크 입력
│   ├── Controls.tsx       # 컨트롤 UI
│   ├── StatusBar.tsx      # 상태 표시
│   └── ConversationPanel.tsx # 대화 기록
├── hooks/           # Custom hooks
│   ├── useLiveKit.ts      # LiveKit 연결
│   ├── useWebSocket.ts    # WebSocket 연결
│   ├── useAudioRecorder.ts # 오디오 녹음
│   └── useAvatarSession.ts # 세션 관리
├── lib/             # 유틸리티
│   └── api.ts             # API 클라이언트
├── types/           # TypeScript 타입
│   └── index.ts
├── App.tsx          # 메인 앱
├── main.tsx         # 엔트리포인트
└── index.css        # 전역 스타일
```

## 주요 기능

### 아바타 뷰
- WebRTC 비디오 스트리밍 (LiveKit)
- WebSocket 프레임 렌더링 (폴백)
- 감정 상태 표시
- 파이프라인 상태 인디케이터

### 오디오 레코더
- 실시간 마이크 입력
- 오디오 레벨 시각화
- VAD (Voice Activity Detection) 지원

### 컨트롤
- 연결/종료 버튼
- 음소거 토글
- 전체화면 모드
- 감정 선택기

### 대화 패널
- 실시간 대화 기록
- 현재 음성 인식 텍스트 표시
- 타임스탬프 및 감정 정보

## 커스터마이징

### 테마 변경
`tailwind.config.js`에서 색상 및 스타일 수정:

```js
theme: {
  extend: {
    colors: {
      primary: { ... },
      avatar: { ... },
    },
  },
}
```

### 새 컴포넌트 추가
1. `src/components/`에 컴포넌트 생성
2. `src/components/index.ts`에 export 추가
3. 필요한 곳에서 import하여 사용

## API 연동

백엔드 API 엔드포인트:

- `GET /health` - 서버 상태
- `POST /api/generate-token` - LiveKit 토큰
- `POST /api/avatar/create` - 세션 생성
- `WS /ws/avatar` - 실시간 통신
