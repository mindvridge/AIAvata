# 서버 재시작 가이드

## 서버 재시작 완료

✅ **서버가 재시작되었습니다!**

- 포트 8000에서 실행 중
- 백그라운드 프로세스로 실행

## 서버 상태 확인

### 1. 포트 확인
```powershell
Get-NetTCPConnection -LocalPort 8000 -State Listen
```

### 2. Health Check
```powershell
curl http://localhost:8000/health
```

### 3. 서버 로그 확인
서버가 백그라운드에서 실행 중이므로, 로그는 콘솔에 출력됩니다.

## 립싱크 테스트

서버가 시작된 후:
1. 프론트엔드 연결 (http://localhost:3000)
2. 아바타 연결
3. 음성 입력 또는 채팅 메시지 전송
4. 립싱크 동작 확인

로그에서 다음 메시지를 확인하세요:
- `🎬 Starting audio stream rendering: ... MuseTalk available=True`
- `🎤 Applying MuseTalk lip sync: ...`
- `✅ MuseTalk lip sync successful: ...`

## 문제 해결

서버가 시작되지 않으면:
1. 포트 8000이 이미 사용 중인지 확인
2. Python 프로세스 확인 및 종료
3. 로그 확인하여 오류 메시지 확인

