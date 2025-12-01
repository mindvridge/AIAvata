# WebSocket 연결 가이드

## 연결 상태 확인

✅ **WebSocket이 성공적으로 연결되었습니다!**

### 연결 정보
- 세션 ID: `a4034c4f-40ec-4c18-ae5b-4969c93707d1`
- WebSocket URL: `ws://localhost:8000/ws/avatar/a4034c4f-40ec-4c18-ae5b-4969c93707d1`
- 상태: Connected

## 다음 단계

### 1. Idle 프레임 확인
연결 직후 서버는 idle 프레임 스트림을 전송합니다.
- 브라우저 콘솔에서 "Received video frame" 메시지 확인
- 아바타 뷰에 프레임이 표시되는지 확인

### 2. 립싱크 테스트

#### 방법 1: 음성 입력 (마이크)
1. 마이크 버튼 클릭
2. 말하기 시작
3. 립싱크 동작 확인

#### 방법 2: 채팅 메시지
1. 채팅 입력창에 메시지 입력
2. Enter 키 또는 전송 버튼 클릭
3. 아바타가 응답하면서 립싱크 동작 확인

### 3. 로그 확인

#### 브라우저 콘솔에서 확인할 메시지:
```
✅ Received video frame, size: [크기]
✅ App: Received video frame, size: [크기]
```

#### 서버 로그에서 확인할 메시지 (개발자 도구 Network 탭 또는 서버 콘솔):
```
🎬 Starting audio stream rendering: sample_rate=24000, fps=30, bytes_per_frame=1600, MuseTalk available=True
🎤 Applying MuseTalk lip sync: frame shape=(512, 512, 3), audio samples=800
✅ MuseTalk lip sync successful: output shape=(512, 512, 3)
```

## 문제 해결

### 프레임이 수신되지 않는 경우
1. 브라우저 콘솔에서 오류 메시지 확인
2. 서버 로그에서 오류 확인
3. WebSocket 연결 상태 확인 (연결 해제되었는지)

### 립싱크가 작동하지 않는 경우
1. 서버 로그에서 MuseTalk 관련 메시지 확인
2. "⚠️ Using lip sync simulation" 메시지가 보이면 MuseTalk이 fallback 상태
3. "✅ MuseTalk lip sync successful" 메시지가 보이면 정상 작동

### 연결이 끊기는 경우
1. 네트워크 상태 확인
2. 서버가 실행 중인지 확인
3. 자동 재연결이 시도되는지 확인

## 참고

- WebSocket 연결은 자동 재연결 기능이 활성화되어 있습니다
- 연결이 끊기면 3초 후 자동으로 재연결 시도
- 립싱크는 오디오 데이터가 있을 때만 작동합니다

