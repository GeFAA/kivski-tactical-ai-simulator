@echo off
REM Kivski Tactical AI Simulator - Doppelklick-Launcher fuer Windows.
REM Ruft start.ps1 mit gelockerter Execution-Policy auf.

cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 (
    echo.
    echo [!] Start fehlgeschlagen. Druecke eine Taste zum Schliessen ...
    pause >nul
)
