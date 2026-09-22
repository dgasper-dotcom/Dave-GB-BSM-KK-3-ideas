@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if not defined OUT set "OUT=%ROOT%"
if not defined POLL_SECONDS set "POLL_SECONDS=600"
if not defined ORDERBOOK_WORKERS set "ORDERBOOK_WORKERS=1"
if not defined CONTRACTS set "CONTRACTS=100"
if not defined MIN_CONTRACTS set "MIN_CONTRACTS=1"
if not defined MAX_ASK set "MAX_ASK=0.99"
if not defined KNOWN_OUTCOME_APY set "KNOWN_OUTCOME_APY=0.0325"
if not defined KNOWN_OUTCOME_LOOKBACK_DAYS set "KNOWN_OUTCOME_LOOKBACK_DAYS=3"
if not defined KNOWN_OUTCOME_MIN_NET_PROFIT_PER_CONTRACT set "KNOWN_OUTCOME_MIN_NET_PROFIT_PER_CONTRACT=0.001"

set "PYTHON=%ROOT%.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo Known outcome loop start. Poll seconds: %POLL_SECONDS%

:loop
echo [%DATE% %TIME%] known-outcome-scan start>> "%ROOT%known_outcome_loop.log"

set "EXTRA_ARGS="
if defined KNOWN_OUTCOME_SERIES_TICKER set "EXTRA_ARGS=%EXTRA_ARGS% --known-outcome-series-ticker %KNOWN_OUTCOME_SERIES_TICKER%"
if defined MAX_MARKET_PAGES set "EXTRA_ARGS=%EXTRA_ARGS% --max-market-pages %MAX_MARKET_PAGES%"

"%PYTHON%" "%ROOT%run_known_outcome.py" scan ^
  --output-dir "%OUT%" ^
  --contracts "%CONTRACTS%" ^
  --min-contracts "%MIN_CONTRACTS%" ^
  --max-ask "%MAX_ASK%" ^
  --known-outcome-apy "%KNOWN_OUTCOME_APY%" ^
  --known-outcome-lookback-days "%KNOWN_OUTCOME_LOOKBACK_DAYS%" ^
  --known-outcome-min-net-profit-per-contract "%KNOWN_OUTCOME_MIN_NET_PROFIT_PER_CONTRACT%" ^
  --known-outcome-orderbook-workers "%ORDERBOOK_WORKERS%" ^
  %EXTRA_ARGS% >> "%ROOT%known_outcome_loop.log" 2>&1

if errorlevel 1 (
  echo [%DATE% %TIME%] known-outcome-scan failed>> "%ROOT%known_outcome_loop.log"
) else (
  echo [%DATE% %TIME%] known-outcome-scan complete>> "%ROOT%known_outcome_loop.log"
)

timeout /t %POLL_SECONDS% /nobreak >nul
goto loop
