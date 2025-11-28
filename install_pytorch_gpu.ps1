# PyTorch CUDA 12.1 설치 스크립트
# NVIDIA GeForce RTX 5060 Ti (CUDA 13.0 지원)에 최적화
# CUDA 13.0은 CUDA 12.1과 호환되므로 PyTorch CUDA 12.1 버전 사용

Write-Host "========================================" -ForegroundColor Green
Write-Host "PyTorch GPU (CUDA 12.1) 설치" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 현재 PyTorch 버전 확인
Write-Host "현재 PyTorch 상태 확인 중..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
Write-Host ""

# PyTorch CUDA 12.1 설치
Write-Host "PyTorch CUDA 12.1 버전 설치 중..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host ""
Write-Host "설치 완료! 확인 중..." -ForegroundColor Yellow
python -c "import torch; print(''); print('='*50); print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); if torch.cuda.is_available(): print(f'CUDA version: {torch.version.cuda}'); print(f'GPU name: {torch.cuda.get_device_name(0)}'); print(f'GPU count: {torch.cuda.device_count()}'); print('='*50)"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "설치 성공!" -ForegroundColor Green
    Write-Host "이제 GPU를 사용할 수 있습니다." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "설치 중 오류가 발생했습니다." -ForegroundColor Red
    Write-Host "수동으로 다음 명령어를 실행해보세요:" -ForegroundColor Yellow
    Write-Host "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Cyan
}

