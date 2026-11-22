@echo off
rem Starts Voice Typing in the background (no window). Use run_console.bat to see errors.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" voice_typing.py
