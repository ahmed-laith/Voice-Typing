@echo off
setlocal enabledelayedexpansion
title Voice Typing - One-Click Setup
cd /d "%~dp0"

echo ===================================================
echo           Voice Typing - تثبيت أداة الإملاء الذكية
echo ===================================================
echo.

:: 1. Check Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo يرجى تثبيت Python 3.10 أو أحدث وتفعيل Add Python to PATH.
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

:: 2. Create Virtual Environment if not exists
if not exist ".venv\Scripts\python.exe" (
    echo [*] إنشاء البيئة الافتراضية .venv...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] فشل إنشاء البيئة الافتراضية.
        pause
        exit /b 1
    )
)

:: 3. Install Requirements
echo [*] تثبيت الحزم والمكتبات المطلوبة...
.venv\Scripts\python.exe -m pip install --upgrade pip --quiet
.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet
if %ERRORLEVEL% neq 0 (
    echo [ERROR] حدث خطأ أثناء تثبيت الحزم.
    pause
    exit /b 1
)

:: 4. Create Desktop & Start Menu Shortcuts
echo [*] إنشاء اختصارات سطح المكتب وقائمة ابدأ...
powershell -NoProfile -ExecutionPolicy Bypass -File "install_shortcuts.ps1"

echo.
echo ===================================================
echo             تم التثبيت بنجاح! اكتمل الإعداد.
echo ===================================================
echo.
echo يمكنك الآن تشغيل الأداة باختصار F1 أو من أيقونة سطح المكتب.
echo.
pause

:: Launch app
start "" ".venv\Scripts\pythonw.exe" voice_typing.py --show
