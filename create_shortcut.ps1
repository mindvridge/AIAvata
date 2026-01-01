# AI Avatar 서비스 바로가기 생성 스크립트

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\AI Avatar 서비스.lnk")
$Shortcut.TargetPath = "$PSScriptRoot\run.bat"
$Shortcut.WorkingDirectory = $PSScriptRoot
$Shortcut.Description = "AI Avatar 서비스 자동 실행"
$Shortcut.IconLocation = "C:\Windows\System32\cmd.exe,0"
$Shortcut.Save()

Write-Host "바로가기가 바탕화면에 생성되었습니다!" -ForegroundColor Green
Write-Host "이제 더블클릭만 하면 서버가 자동으로 시작됩니다." -ForegroundColor Green

