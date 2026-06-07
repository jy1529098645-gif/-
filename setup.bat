@echo off
REM 量化研究工具 · 一键安装（建 venv + 装依赖 + 跑测试 + 预热数据）
cd /d "%~dp0"
echo === [1/4] 创建虚拟环境 ===
if not exist .venv ( python -m venv .venv )
echo === [2/4] 安装依赖 ===
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo === [3/4] 运行测试 ===
".venv\Scripts\python.exe" -m pytest -q
echo === [4/4] 预热数据缓存（SPY + 七姐妹 + 宏观 + 财报）===
".venv\Scripts\python.exe" scripts\warmup.py
echo.
echo 安装完成！运行前端： run_app.bat   （或 .venv\Scripts\streamlit run app.py）
pause
