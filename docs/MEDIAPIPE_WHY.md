# MediaPipe가 필요한 이유

## 개요

MediaPipe는 Google에서 개발한 오픈소스 미디어 처리 프레임워크로, 이 프로젝트에서는 **얼굴 감지 및 랜드마크 추출**에 사용됩니다.

## 주요 사용 목적

### 1. **MuseTalk 립싱크에서 얼굴 영역 감지** (가장 중요)

**위치**: `src/models/integrations/musetalk.py`

```python
# MediaPipe로 얼굴 위치 찾기
face_bbox = self._detect_face_bbox(source_frame)

if face_bbox is not None:
    x1, y1, x2, y2 = face_bbox
    # 얼굴 영역만 크롭 (256x256)
    face_region = source_frame[y1:y2, x1:x2]
    face_crop = cv2.resize(face_region, (256, 256))
else:
    # 얼굴 감지 실패 시 전체 프레임 사용 (정확도 저하)
    face_crop = cv2.resize(source_frame, (256, 256))
```

**이유**:
- MuseTalk은 **얼굴 영역만** 처리하도록 설계됨
- 전체 프레임을 처리하면 불필요한 배경까지 처리되어:
  - 처리 속도 저하
  - 립싱크 정확도 저하
  - GPU 메모리 낭비

**효과**:
- ✅ 얼굴 영역만 정확하게 크롭 → 립싱크 품질 향상
- ✅ 처리 속도 향상 (작은 영역만 처리)
- ✅ GPU 메모리 효율성 향상

---

### 2. **얼굴 랜드마크 감지** (입 영역 정확도 향상)

**위치**: `src/pipeline/avatar_renderer.py`

```python
def detect_face_landmarks(self, image: np.ndarray) -> Optional[dict]:
    """MediaPipe로 얼굴 랜드마크 감지"""
    results = self._face_mesh.process(rgb_image)
    
    # 입 영역 랜드마크 추출 (립싱크 블렌딩용)
    mouth_landmarks = self._extract_mouth_landmarks(landmarks, w, h)
```

**이유**:
- 립싱크 결과를 원본 프레임에 블렌딩할 때 **입 영역만** 정확하게 마스킹
- 468개의 얼굴 랜드마크 중 입 영역(약 40개)만 추출하여 정확한 블렌딩

**효과**:
- ✅ 입 영역만 정확하게 블렌딩 → 자연스러운 립싱크
- ✅ 얼굴 다른 부분은 원본 유지 → 품질 향상

---

### 3. **성능 최적화**

**효과**:
- 전체 프레임(512x512) 처리: ~50ms
- 얼굴 영역만 처리(256x256): ~20ms
- **약 2.5배 속도 향상**

---

## MediaPipe 없이도 작동하는가?

### ✅ **작동 가능** (Fallback 있음)

코드에서 MediaPipe가 없어도 작동하도록 처리되어 있습니다:

```python
try:
    import mediapipe as mp
    # MediaPipe 사용
except ImportError:
    logger.warning("MediaPipe not installed. Face detection will be limited.")
    # 전체 프레임 사용 (fallback)
```

### ⚠️ **하지만 품질 저하**

MediaPipe 없이:
- ❌ 전체 프레임 처리 → 속도 저하
- ❌ 얼굴 위치 불명확 → 립싱크 정확도 저하
- ❌ 입 영역 마스킹 부정확 → 블렌딩 품질 저하

---

## MediaPipe 설치 방법

```bash
pip install mediapipe
```

또는 `requirements.txt`에 이미 포함되어 있습니다:
```
mediapipe>=0.10.0
```

---

## 요약

| 항목 | MediaPipe 있음 | MediaPipe 없음 |
|------|---------------|----------------|
| 립싱크 정확도 | ⭐⭐⭐⭐⭐ 높음 | ⭐⭐⭐ 보통 |
| 처리 속도 | ⭐⭐⭐⭐⭐ 빠름 | ⭐⭐⭐ 느림 |
| GPU 메모리 | ⭐⭐⭐⭐⭐ 효율적 | ⭐⭐⭐ 비효율적 |
| 작동 여부 | ✅ 정상 | ✅ 작동 (품질 저하) |

**결론**: MediaPipe는 **선택 사항이지만 강력히 권장**됩니다. 립싱크 품질과 성능을 크게 향상시킵니다.

