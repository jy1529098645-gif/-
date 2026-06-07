@echo off
REM 量化研究工具 · 本地前端启动器
cd /d "%~dp0"
".venv\Scripts\streamlit.exe" run app.py
pause
