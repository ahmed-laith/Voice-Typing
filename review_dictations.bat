@echo off
rem Opens the dictation review page: listen to what you dictated, fix the text, add correction rules.
cd /d "%~dp0"
".venv\Scripts\python.exe" recorder.py --review
