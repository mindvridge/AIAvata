# 백엔드 서버 재시작 스크립트

Write-Host "========================================" -ForegroundColor Green
Write-Host "백엔드 서버 재시작" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# 포트 8000을 사용하는 프로세스 확인 및 종료
Write-Host "기존 서버 프로세스 확인 중..." -ForegroundColor Yellow
$processes = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique

if ($processes) {
    foreach ($pid in $processes) {
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "프로세스 종료 중: $($proc.ProcessName) (PID: $pid)" -ForegroundColor Yellow
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 2
        }
    }
} else {
    Write-Host "실행 중인 서버가 없습니다." -ForegroundColor Cyan
}

Write-Host ""
Write-Host "백엔드 서버 시작 중..." -ForegroundColor Yellow
Write-Host ""

# 백엔드 서버 시작
# 개발 모드로 실행 (--reload 옵션 포함)
Start-Process -FilePath "python" -ArgumentList "-m", "uvicorn", "src.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000" -WindowStyle Normal

Write-Host "서버가 시작되었습니다." -ForegroundColor Green
Write-Host "로그를 확인하려면 새로 열린 창을 확인하세요." -ForegroundColor Cyan
Write-Host ""
Write-Host "서버 상태 확인: http://localhost:8000/health" -ForegroundColor Cyan

