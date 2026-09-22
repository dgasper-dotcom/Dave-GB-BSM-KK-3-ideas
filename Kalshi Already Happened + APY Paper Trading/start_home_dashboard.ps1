param()

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$DashboardHost = if ($env:HOST) { $env:HOST } else { "127.0.0.1" }
$Port = if ($env:PORT) { [int]$env:PORT } else { 8766 }
$env:CONTRACTS = if ($env:CONTRACTS) { $env:CONTRACTS } else { "100" }
$env:POLL_SECONDS = if ($env:POLL_SECONDS) { $env:POLL_SECONDS } else { "600" }
$env:ORDERBOOK_WORKERS = if ($env:ORDERBOOK_WORKERS) { $env:ORDERBOOK_WORKERS } else { "1" }
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

Set-Location $Root

Start-Process powershell.exe -WorkingDirectory $Root -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", (Join-Path $Root "known_outcome_loop.ps1")
)

$dashboardLog = Join-Path $Root "dashboard.log"
$dashboardCommand = "& '$Python' '$Root\run_dashboard.py' --host '$DashboardHost' --port '$Port' --output-dir '$Root' *>> '$dashboardLog'"
Start-Process powershell.exe -WorkingDirectory $Root -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $dashboardCommand
)

Write-Host "Known-outcome scanner started in a separate PowerShell window."
Write-Host "Dashboard: http://localhost:$Port"
Write-Host "Dashboard started in a separate PowerShell window."
