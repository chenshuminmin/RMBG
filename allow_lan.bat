@echo off
REM Allow same-WiFi devices to reach the RMBG service on TCP 6574.
REM Double-click this file and accept the Windows UAC prompt (admin required for firewall rules).

cd /d "%~dp0"
net session >nul 2>&1
if errorlevel 1 (
  echo Requesting administrator privileges, please click "Yes" on the UAC prompt...
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $n = 'RMBG WhiteBg TCP 6574'; Get-NetFirewallRule -DisplayName $n -ErrorAction SilentlyContinue | Remove-NetFirewallRule; New-NetFirewallRule -DisplayName $n -Direction Inbound -Protocol TCP -LocalPort 6574 -Action Allow -Profile Any -RemoteAddress LocalSubnet -Description 'Allow LAN clients to reach RMBG white-background API' | Out-Null; $r = Get-NetFirewallRule -DisplayName $n; Write-Host ('Rule: ' + $r.DisplayName + ' | Enabled=' + $r.Enabled + ' | Action=' + $r.Action); Write-Host '--- listening ---'; Get-NetTCPConnection -LocalPort 6574 -State Listen -ErrorAction SilentlyContinue | Select-Object LocalAddress,LocalPort | Format-Table -AutoSize | Out-String | Write-Host; Write-Host 'Laptop URL: http://192.168.1.14:6574/health' }"

echo.
pause
