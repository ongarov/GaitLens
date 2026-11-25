@echo off
cd /d "%~dp0"
echo Starting GaitLens... Do not close this window. Browser will open automatically.
python -m streamlit run app.py
pause
