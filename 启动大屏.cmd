@echo off
setlocal
cd /d "%~dp0"
if exist "..\work\venv\Scripts\python.exe" (
    "..\work\venv\Scripts\python.exe" -m streamlit run app.py --server.address localhost --server.headless false
) else (
    python -m streamlit run app.py --server.address localhost --server.headless false
)
if errorlevel 1 pause
