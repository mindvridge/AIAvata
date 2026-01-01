# AI Avatar 서비스 바로가기 생성

$batPath = Join-Path $PSScriptRoot "run.bat"
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "AI Avatar 서비스.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)
$Shortcut.TargetPath = $batPath
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.Description = "AI Avatar 서비스 자동 실행"
$Shortcut.Save()

Write-Host "바로가기가 바탕화면에 생성되었습니다!" -ForegroundColor Green
Write-Host "경로: $shortcutPath" -ForegroundColor Cyan

