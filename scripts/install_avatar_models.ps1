# MuseTalk & LivePortrait 설치 스크립트
# 실행 전 GPU (NVIDIA CUDA)가 설치되어 있어야 합니다.

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  MuseTalk & LivePortrait 설치 스크립트" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. 모델 디렉토리 생성
Write-Host "[1/5] 모델 디렉토리 생성..." -ForegroundColor Yellow
$modelDir = "models"
$musetalkDir = "$modelDir/musetalk"
$livePortraitDir = "$modelDir/live_portrait"

New-Item -ItemType Directory -Force -Path $musetalkDir | Out-Null
New-Item -ItemType Directory -Force -Path $livePortraitDir | Out-Null

Write-Host "  - $musetalkDir 생성됨" -ForegroundColor Green
Write-Host "  - $livePortraitDir 생성됨" -ForegroundColor Green

# 2. 필수 패키지 설치
Write-Host ""
Write-Host "[2/5] 필수 Python 패키지 설치..." -ForegroundColor Yellow

# MuseTalk 의존성
pip install diffusers accelerate transformers

# 얼굴 파싱/감지
pip install face-alignment dlib

# 오디오 처리
pip install librosa soundfile

Write-Host "  기본 패키지 설치 완료" -ForegroundColor Green

# 3. MuseTalk 설치 안내
Write-Host ""
Write-Host "[3/5] MuseTalk 설치..." -ForegroundColor Yellow
Write-Host ""
Write-Host "MuseTalk은 수동 설치가 필요합니다:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. GitHub 저장소 클론:" -ForegroundColor White
Write-Host "     git clone https://github.com/TMElyralab/MuseTalk.git" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. 모델 다운로드:" -ForegroundColor White
Write-Host "     - Hugging Face에서 모델 다운로드" -ForegroundColor Gray
Write-Host "     - https://huggingface.co/TMElyralab/MuseTalk" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. 모델 파일을 models/musetalk/에 복사" -ForegroundColor White
Write-Host ""

# 4. LivePortrait 설치 안내
Write-Host "[4/5] LivePortrait 설치..." -ForegroundColor Yellow
Write-Host ""
Write-Host "LivePortrait은 수동 설치가 필요합니다:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  1. GitHub 저장소 클론:" -ForegroundColor White
Write-Host "     git clone https://github.com/KwaiVGI/LivePortrait.git" -ForegroundColor Gray
Write-Host ""
Write-Host "  2. 의존성 설치:" -ForegroundColor White
Write-Host "     cd LivePortrait" -ForegroundColor Gray
Write-Host "     pip install -r requirements.txt" -ForegroundColor Gray
Write-Host ""
Write-Host "  3. 모델 다운로드 (자동):" -ForegroundColor White
Write-Host "     python download_models.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  4. 모델 파일을 models/live_portrait/에 복사" -ForegroundColor White
Write-Host ""

# 5. 간단한 대안 - SadTalker
Write-Host "[5/5] 간단한 대안: SadTalker" -ForegroundColor Yellow
Write-Host ""
Write-Host "더 쉬운 설치를 원하시면 SadTalker를 사용할 수 있습니다:" -ForegroundColor Cyan
Write-Host ""
Write-Host "  pip install sadtalker" -ForegroundColor Gray
Write-Host ""
Write-Host "또는 Hugging Face Spaces에서 온라인으로 테스트:" -ForegroundColor White
Write-Host "  https://huggingface.co/spaces/vinthony/SadTalker" -ForegroundColor Gray
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  설치 스크립트 완료" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "참고:" -ForegroundColor Yellow
Write-Host "  - GPU 메모리: 최소 8GB VRAM 권장" -ForegroundColor White
Write-Host "  - CUDA 버전: 11.8 이상 권장" -ForegroundColor White
Write-Host "  - 모델 다운로드: 약 2-5GB 필요" -ForegroundColor White
Write-Host ""

