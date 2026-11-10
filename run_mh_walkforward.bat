@echo off
rem Starts the lean monthly-high walk-forward app (opens in your browser). Close this window to stop it.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m streamlit run mh_walkforward_ui.py
