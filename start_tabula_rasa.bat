@echo off
title Tabula Rasa AI - Startup
cd /d "C:\Users\Admin\tabula-rasa"
cls

set LOGFILE=C:\Users\Admin\tabula-rasa\startup_debug.log
echo === Tabula Rasa Startup %date% %time% === > "%LOGFILE%"
echo Working directory: %cd% >> "%LOGFILE%"

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║        Tabula Rasa AI System Startup        ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ─── Kill old processes ───
echo [*] Cleaning up old processes...
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8000 " ^| find "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| find ":8002 " ^| find "LISTENING"') do taskkill /F /PID %%a >nul 2>&1
ping -n 4 127.0.0.1 >nul
echo  [+] Ports 8000,8002 cleared.
echo.

:: ─── Start Tabula Rasa AI (Port 8002) ───
echo [*] Starting Tabula Rasa AI (port 8002)...
start "Tabula Rasa AI (8002)" /MIN cmd /c "cls & title Tabula Rasa AI (8002) & python -c "from egefalos.tabula_rasa import main; main()""
echo  [+] Window opened (minimized).
echo.

:: ─── Start API Server (Port 8000) ───
echo [*] Starting API Server (port 8000)...
start "API Server (8000)" /MIN cmd /c "cls & title API Server (8000) & python scripts\api_server.py"
echo  [+] Window opened (minimized).
echo.

:: ─── Wait ───
echo [*] Waiting 20 seconds for models to load...
ping -n 21 127.0.0.1 >nul
echo  [+] Wait complete.
echo.

:: ─── Open dashboard ───
echo [*] Opening dashboard in browser...
start http://localhost:8000/
echo  [+] Dashboard opened.
echo.

:: ─── Health check ───
echo === Health Check === >> "%LOGFILE%"
ping -n 4 127.0.0.1 >nul
python -c "
import urllib.request
for url, name in [('http://127.0.0.1:8000/health','8000 health'),('http://127.0.0.1:8000/','8000 root'),('http://127.0.0.1:8002/skills','8002 skills')]:
    try:
        r = urllib.request.urlopen(url, timeout=3)
        print('[OK] %s - %d bytes' % (name, len(r.read())))
    except Exception as e:
        print('[FAIL] %s - %s' % (name, e))
" >> "%LOGFILE%" 2>&1

echo. >> "%LOGFILE%"
echo === Servers running in separate windows === >> "%LOGFILE%"
echo.

echo  ╔══════════════════════════════════════════════╗
echo  ║           All systems running!              ║
echo  ║                                            ║
echo  ║  Dashboard      : http://localhost:8000    ║
echo  ║  Tabula Rasa AI : http://localhost:8002    ║
echo  ║                                            ║
echo  ║  Check startup_debug.log if issues.        ║
echo  ║                                            ║
echo  ║  Close each server window to stop.         ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo [*] Server windows are minimized — bring them up to see output.
echo [*] Each window starts with a clean screen.
echo [*] Press any key to close this launcher (servers keep running).
pause >nul
