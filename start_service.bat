@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM 白底图服务（Windows 原生启动，等价于 docker compose 暴露的 6574 端口）
REM 首次运行前请确认：.venv 已装好 requirements.txt + CUDA 版 torch，resources\BiRefNet-HRSOD 已下好模型

set "PORT=6574"
set "MODEL_PATH=%~dp0resources\BiRefNet-HRSOD"

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] 未找到虚拟环境 .venv，请先执行: python -m venv .venv
  pause
  exit /b 1
)

echo Starting rmbg service on http://127.0.0.1:%PORT%
echo MODEL_PATH=%MODEL_PATH%
".venv\Scripts\python.exe" server.py
