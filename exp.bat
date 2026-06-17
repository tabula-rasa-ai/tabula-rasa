@echo off
title Tabula Rasa Experiment Runner
cd /d "%~dp0"

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║     Tabula Rasa — Complete Experiment       ║
echo  ╚══════════════════════════════════════════════╝
echo.
echo  One command: train + benchmark + probe + export + log
echo.
echo  Examples:
echo    exp --name baseline --preset 1M --steps 10000
echo    exp --name moe_test --moe --preset 5M --steps 20000
echo    exp --name full_rasa --rasa --moe --preset 10M --steps 50000
echo    exp --list
echo.

python scripts\run_experiment.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  [!] Experiment failed. Check output above.
    pause
)
