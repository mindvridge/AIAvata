# 채팅 립싱크 문제 수정

## 문제 상황
채팅 메시지를 보냈을 때 립싱크가 작동하지 않는 문제가 있었습니다.

## 원인 분석
1. **Idle 스트림과 충돌**: 연결 직후 시작된 idle 스트림이 `is_streaming` 플래그를 True로 설정하여 채팅 립싱크 처리를 막았습니다.
2. **스트림 중지 로직 부재**: 채팅 처리 시작 전에 idle 스트림을 명시적으로 중지하지 않았습니다.

## 수정 사항

### 1. Idle 스트림 관리 개선
- `idle_task`를 connection 정보에 저장
- 채팅 처리 시작 전 idle 스트림 명시적으로 중지
- 립싱크 완료 후 idle 스트림 자동 재시작

### 2. 스트림 상태 관리
- `is_streaming` 플래그 충돌 해결
- `processing_audio` 플래그로 오디오 처리 중임을 명시
- Idle 스트림이 오디오 처리 중에는 프레임 건너뛰기

### 3. 로그 개선
- 립싱크 시작/완료 시 명확한 로그 메시지 추가
- 프레임 전송 진행 상황 로그

## 테스트 방법

1. 채팅 메시지 입력 (예: "안녕하세요")
2. 서버 로그 확인:
   ```
   ⏸️ Stopping idle stream for chat processing...
   🎤 Starting chat TTS and lip sync processing...
   🎬 Starting lip sync rendering: ...
   📹 Sent 30 lip sync frames
   ✅ Lipsync video stream completed: ... frames sent
   ```
3. 아바타 립싱크 동작 확인

## 변경된 파일
- `src/api/websocket.py` - Idle 스트림 관리 및 채팅 처리 로직 개선

