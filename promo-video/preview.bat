@echo off
cd /d "%~dp0"
if exist "D:\AHMED\Ai\Projects\habibi-voice\out\20261122-192812\*.wav" (
    copy /y "D:\AHMED\Ai\Projects\habibi-voice\out\20261122-192812\*.wav" "audio\" >nul 2>&1
)
start "" "index.html"
