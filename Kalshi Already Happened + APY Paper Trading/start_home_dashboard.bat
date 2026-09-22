@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=8766"
if not defined CONTRACTS set "CONTRACTS=100"
if not defined POLL_SECONDS set "POLL_SECONDS=600"
if not defined ORDERBOOK_WORKERS set "ORDERBOOK_WORKERS=1"
if not defined MAX_MARKET_PAGES set "MAX_MARKET_PAGES=1"
if not defined KNOWN_OUTCOME_SERIES_TICKER set "KNOWN_OUTCOME_SERIES_TICKER=KXMLBHIT,KXMLBHR,KXTEMPNYCH,KXTEMPCHIH,KXTEMPDCH,KXTEMPLAXH,KXTEMPMIAH,KXTEMPAUSH"

set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

start "Kalshi known outcome scanner" cmd /k ""%ROOT%known_outcome_loop.bat""
start "Kalshi dashboard" cmd /k ""%PYTHON%" "%ROOT%run_dashboard.py" --host "%HOST%" --port "%PORT%" --output-dir "%ROOT%" >> "%ROOT%dashboard.log" 2>&1"

echo Known-outcome scanner started in a new Command Prompt window.
echo Dashboard: http://localhost:%PORT%
echo Dashboard started in a new Command Prompt window.
