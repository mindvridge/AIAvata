# TTS 디버깅 가이드

## 문제 현상
- TTS 로그가 안보임
- 파동 그래프 작동 안함

## 확인 사항

### 1. 메시지 수신 확인
서버 로그에서 다음 메시지가 나타나야 합니다:
```
📥 Raw message received: type=websocket.receive
📝 Received text message: {"type":"chat","text":"..."}
📩 Processing control message type: chat
💬 Chat message received in _handle_control: '...'
🚀 Calling _handle_chat with text: '...'
```

### 2. 채팅 처리 확인
```
Chat message received: ...
LLM response: ...
🎤 Starting TTS and lip sync processing for chat response...
```

### 3. TTS 처리 확인
```
🎙️ Calling Chatterbox TTS synthesize: text='...', voice_id=...
✅ TTS synthesize completed: ... samples
🔊 TTS audio generated: ... samples (... bytes)
📤 Sending audio_data message to frontend: ... chars (base64)
✅ audio_data message sent successfully
```

## 문제 해결

### 채팅 메시지가 서버에 도달하지 않는 경우
- 프론트엔드 콘솔에서 WebSocket 연결 상태 확인
- `sendChat` 함수가 호출되는지 확인
- WebSocket 메시지 전송 로그 확인

### TTS 로그가 없는 경우
- `_handle_chat` 함수가 호출되는지 확인
- LLM 응답 생성이 완료되는지 확인
- `_process_chat_with_tts` 함수가 호출되는지 확인

### 오디오가 생성되지 않는 경우
- Chatterbox TTS 모델이 초기화되었는지 확인
- Mock 오디오가 생성되는지 확인 (최소한 파동은 보여야 함)
- 오류 메시지 확인


