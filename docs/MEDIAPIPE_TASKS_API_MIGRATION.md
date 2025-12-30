# MediaPipe Tasks API 마이그레이션 완료

## 개요

MediaPipe 0.10.31에서는 `solutions` API가 제거되고 새로운 `tasks` API를 사용합니다. 코드를 업데이트하여 두 API를 모두 지원하도록 했습니다.

## 변경 사항

### 1. **AvatarRenderer** (`src/pipeline/avatar_renderer.py`)

#### Face Mesh 초기화
- **이전**: `mp.solutions.face_mesh.FaceMesh()` 만 사용
- **이후**: 
  - `solutions` API가 있으면 → 사용
  - 없으면 → `tasks` API의 `FaceLandmarker` 사용

```python
# tasks API 사용 예시
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import vision_task_running_mode

base_opts = base_options.BaseOptions(
    model_asset_path=None,  # 번들된 모델 사용
    delegate=base_options.BaseOptions.Delegate.CPU
)
options = vision.FaceLandmarkerOptions(
    base_options=base_opts,
    num_faces=1,
    min_face_detection_confidence=0.5,
    running_mode=vision_task_running_mode.VisionTaskRunningMode.VIDEO
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)
```

#### 얼굴 랜드마크 감지
- `detect_face_landmarks()`: 두 API 모두 지원
- `detect_mouth_region()`: 두 API 모두 지원

### 2. **MuseTalk** (`src/models/integrations/musetalk.py`)

#### 얼굴 감지
- **이전**: `mp.solutions.face_detection.FaceDetection()` 만 사용
- **이후**: 
  - `solutions` API가 있으면 → 사용
  - 없으면 → `tasks` API의 `FaceDetector` 사용

```python
# tasks API 사용 예시
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import vision_task_running_mode, image as mp_image

base_opts = base_options.BaseOptions(
    model_asset_path=None,
    delegate=base_options.BaseOptions.Delegate.CPU
)
options = vision.FaceDetectorOptions(
    base_options=base_opts,
    running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
    min_detection_confidence=0.5
)
face_detector = vision.FaceDetector.create_from_options(options)

# 이미지 처리
mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
detection_result = face_detector.detect(mp_img)
```

## API 차이점

### Solutions API (구버전)
```python
import mediapipe as mp

face_mesh = mp.solutions.face_mesh.FaceMesh(...)
results = face_mesh.process(rgb_image)
landmarks = results.multi_face_landmarks[0].landmark
```

### Tasks API (신버전)
```python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core import image as mp_image

face_landmarker = vision.FaceLandmarker.create_from_options(options)
mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_image)
result = face_landmarker.detect_for_video(mp_img, timestamp_ms)
landmarks = result.face_landmarks[0]
```

## 주요 변경점

1. **이미지 형식**: `numpy array` → `mp_image.Image` 객체
2. **결과 구조**: `results.multi_face_landmarks` → `result.face_landmarks`
3. **비디오 모드**: `detect_for_video()` 사용 시 timestamp 필요
4. **리소스 관리**: `close()` 메서드로 명시적 해제

## 호환성

- ✅ **MediaPipe 0.9.x 이하**: `solutions` API 사용
- ✅ **MediaPipe 0.10.31**: `tasks` API 사용
- ✅ **Fallback**: API가 없으면 전체 프레임 처리

## 테스트

```bash
# 립싱크 테스트
python test_lipsync_comprehensive.py
```

## 참고

- MediaPipe Tasks API 문서: https://ai.google.dev/edge/mediapipe/solutions/tasks
- Face Landmarker: https://ai.google.dev/edge/mediapipe/solutions/tasks/face_landmarker
- Face Detector: https://ai.google.dev/edge/mediapipe/solutions/tasks/face_detector

