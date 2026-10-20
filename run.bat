@echo off
:: Mundial LLM - Launcher
:: Always uses the local venv Python so all packages are available.
cd /d "%~dp0"
.\venv\Scripts\python -m mundial_llm
pause
