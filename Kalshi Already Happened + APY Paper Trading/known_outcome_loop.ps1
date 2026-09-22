param()

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Out = if ($env:OUT) { $env:OUT } else { $Root }
$DurationMinutes = if ($env:DURATION_MINUTES) { [int]$env:DURATION_MINUTES } else { 20160 }
$PollSeconds = if ($env:POLL_SECONDS) { [int]$env:POLL_SECONDS } else { 600 }
$OrderbookWorkers = if ($env:ORDERBOOK_WORKERS) { [int]$env:ORDERBOOK_WORKERS } else { 1 }
$Contracts = if ($env:CONTRACTS) { [double]$env:CONTRACTS } else { 100 }
$MinContracts = if ($env:MIN_CONTRACTS) { [double]$env:MIN_CONTRACTS } else { 1 }
$MaxAsk = if ($env:MAX_ASK) { [double]$env:MAX_ASK } else { 0.99 }
$KnownOutcomeApy = if ($env:KNOWN_OUTCOME_APY) { [double]$env:KNOWN_OUTCOME_APY } else { 0.0325 }
$KnownOutcomeLookbackDays = if ($env:KNOWN_OUTCOME_LOOKBACK_DAYS) { [int]$env:KNOWN_OUTCOME_LOOKBACK_DAYS } else { 3 }
$KnownOutcomeMinNetProfitPerContract = if ($env:KNOWN_OUTCOME_MIN_NET_PROFIT_PER_CONTRACT) { [double]$env:KNOWN_OUTCOME_MIN_NET_PROFIT_PER_CONTRACT } else { 0.001 }
$LogPath = Join-Path $Root "known_outcome_loop.log"
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

function Write-LoopLog {
    param([string]$Message)
    $stamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    "[$stamp] $Message" | Tee-Object -FilePath $LogPath -Append
}

function Invoke-Scan {
    $args = @(
        (Join-Path $Root "run_known_outcome.py"),
        "scan",
        "--output-dir", $Out,
        "--contracts", "$Contracts",
        "--min-contracts", "$MinContracts",
        "--max-ask", "$MaxAsk",
        "--known-outcome-apy", "$KnownOutcomeApy",
        "--known-outcome-lookback-days", "$KnownOutcomeLookbackDays",
        "--known-outcome-min-net-profit-per-contract", "$KnownOutcomeMinNetProfitPerContract",
        "--known-outcome-orderbook-workers", "$OrderbookWorkers"
    )
    if ($env:KNOWN_OUTCOME_SERIES_TICKER) {
        $args += @("--known-outcome-series-ticker", $env:KNOWN_OUTCOME_SERIES_TICKER)
    }
    if ($env:MAX_MARKET_PAGES) {
        $args += @("--max-market-pages", $env:MAX_MARKET_PAGES)
    }
    & $Python @args 2>&1 | Tee-Object -FilePath $LogPath -Append
    return $LASTEXITCODE
}

Set-Location $Root
$EndAt = (Get-Date).AddMinutes($DurationMinutes)
Write-LoopLog "known outcome loop start duration_minutes=$DurationMinutes poll_seconds=$PollSeconds orderbook_workers=$OrderbookWorkers contracts=$Contracts min_contracts=$MinContracts apy=$KnownOutcomeApy"

while ((Get-Date) -lt $EndAt) {
    Write-LoopLog "known-outcome-scan start"
    $exitCode = Invoke-Scan
    if ($exitCode -eq 0) {
        Write-LoopLog "known-outcome-scan complete"
    } else {
        Write-LoopLog "known-outcome-scan failed exit_code=$exitCode"
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-LoopLog "known outcome loop complete"
