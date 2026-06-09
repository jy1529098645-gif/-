@echo off
REM 每日自动追踪：留痕当日指导 + 回填评估历史判断准确性。可挂 Windows 任务计划(收盘后跑)。
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
if not exist reports mkdir reports
".venv\Scripts\python.exe" -m scripts.daily_track >> "reports\daily_track.log" 2>&1
