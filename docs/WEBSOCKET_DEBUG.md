# WebSocket 연결 디버깅 가이드

## 연결 상태 확인

### 1. 브라우저 콘솔에서 확인할 메시지

**정상 연결 시:**
```
Connecting to WebSocket: ws://localhost:8000/ws/avatar/...
WebSocket connected ✅
Received video frame, size: ... (idle 프레임)
```

**연결 실패 시:**
```
Connecting to WebSocket: ...
WebSocket error: ...
또는
WebSocket closed unexpectedly
```

### 2. 서버 로그에서 확인할 메시지

**연결 성공 시:**
```
WebSocket connection accepted
Starting auto idle stream for connection: ...
Sent ... idle frames
```

**채팅 처리 시:**
```
Chat message received: ...
⏸️ Stopping idle stream for chat processing...
🎤 Starting chat TTS and lip sync processing...
🎬 Starting lip sync rendering: ...
✅ Lipsync video stream completed: ...
```

## 립싱크 작동 확인

1. **연결 확인**: "WebSocket connected" 메시지 확인
2. **Idle 프레임 수신**: "Received video frame" 메시지 확인
3. **채팅 메시지 전송**: 메시지 입력 후 전송
4. **서버 로그 확인**: 위의 립싱크 로그 메시지 확인
5. **비디오 프레임 수신**: 립싱크 프레임이 전송되는지 확인

## 문제 해결

### 연결이 안 되는 경우
- 서버가 실행 중인지 확인
- 포트 8000이 열려있는지 확인
- 브라우저 콘솔 오류 확인

### 립싱크가 작동하지 않는 경우
- 서버 로그에서 "⏸️ Stopping idle stream" 메시지 확인
- "🎤 Starting chat TTS" 메시지 확인
- "🎬 Starting lip sync rendering" 메시지 확인
- 오류 메시지가 있는지 확인


