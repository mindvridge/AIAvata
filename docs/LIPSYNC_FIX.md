# 립싱크 작동 문제 수정

## 문제 상황
MuseTalk 립싱크가 작동하지 않는 문제가 있었습니다.

## 원인 분석
1. **UNet 입력 형식 오류**: 오디오 특징의 형태가 MuseTalk이 기대하는 형식과 맞지 않았습니다.
2. **PositionalEncoding 차원 불일치**: PositionalEncoding이 384 차원을 기대하는데 다른 차원이 전달되었습니다.
3. **오디오 특징 추출 방식**: 실시간 오디오 처리가 MuseTalk의 배치 처리 방식과 달랐습니다.

## 수정 사항

### 1. 오디오 특징 추출 개선
- Whisper encoder의 마지막 레이어 hidden state만 사용 (384 차원)
- 실시간 오디오 버퍼링 및 처리
- PositionalEncoding 입력 형식 맞춤

### 2. UNet 입력 형식 수정
- 오디오 특징을 `[batch, 1, 384]` 형태로 변환
- PositionalEncoding을 통과시킨 후 `[batch, seq_len, hidden_dim]` 형태로 유지
- UNet의 `encoder_hidden_states`로 전달

### 3. UNet 출력 처리
- UNet의 반환값에서 `.sample` 속성 접근
- VAE 디코딩 및 후처리 개선

## 테스트 결과
- ✅ 초기화: 성공
- ✅ 립싱크 처리: 성공 (shape: (256, 256, 3))
- ⚠️ OpenCV resize 경고: 발생하지만 fallback으로 처리됨

## 향후 개선 사항
1. VAE 출력 처리 개선 (OpenCV resize 오류 해결)
2. 실시간 오디오 버퍼링 최적화
3. 여러 프레임의 오디오 특징을 버퍼링하여 MuseTalk 방식과 더 유사하게 처리

## 참고
- MuseTalk의 `get_whisper_chunk`는 여러 프레임의 오디오 특징을 한 번에 생성하지만, 실시간 처리에서는 프레임별로 처리해야 합니다.
- 현재 구현은 단일 프레임 처리에 최적화되어 있으며, 품질 향상을 위해 버퍼링을 추가할 수 있습니다.

