# run_demo.ps1
# ------------
# Starts the fraud detection pipeline, waits for services to be ready,
# opens the Spark Master UI in the browser, then live-tails [FRAUD] lines
# from the producer logs.
#
# Usage:
#   .\run_demo.ps1              # normal start
#   .\run_demo.ps1 -Rebuild     # force rebuild of consumer/producer images first
#   .\run_demo.ps1 -Down        # tear everything down cleanly

param(
    [switch]$Rebuild,
    [switch]$Down
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Force UTF-8 throughout so Docker log bytes (em dashes, Unicode amounts)
# are decoded correctly instead of being mangled into cp1252 garbage (ΓÇö etc.)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding            = [System.Text.Encoding]::UTF8

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Header {
    param([string]$Text)
    $line = "=" * 60
    Write-Host ""
    Write-Host $line              -ForegroundColor Cyan
    Write-Host "  $Text"         -ForegroundColor Cyan
    Write-Host $line              -ForegroundColor Cyan
}

function Write-Step {
    param([string]$Text)
    Write-Host "[>>] $Text" -ForegroundColor Yellow
}

function Write-Ok {
    param([string]$Text)
    Write-Host "[OK] $Text" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Text)
    Write-Host "[!!] $Text" -ForegroundColor Magenta
}

# ---------------------------------------------------------------------------
# Tear-down mode
# ---------------------------------------------------------------------------
if ($Down) {
    Write-Header "Tearing Down Pipeline"
    Write-Step "Stopping and removing all containers and volumes..."
    docker compose down --volumes --remove-orphans
    Write-Ok "Pipeline stopped. Named volumes removed."
    exit 0
}

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
Write-Header "Fraud Detection Pipeline — Demo Runner"
Write-Host "  Press Ctrl+C at any time to stop the log tail." -ForegroundColor Gray
Write-Host "  Run with -Down to stop all containers."         -ForegroundColor Gray

# ---------------------------------------------------------------------------
# Sanity check — Docker must be running
# ---------------------------------------------------------------------------
Write-Step "Checking Docker is running..."
try {
    $null = docker info 2>&1
    if ($LASTEXITCODE -ne 0) { throw "docker info failed" }
    Write-Ok "Docker Desktop is running."
} catch {
    Write-Host "[FAIL] Docker Desktop is not running. Please start it and retry." -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------------------
# Optional rebuild
# ---------------------------------------------------------------------------
if ($Rebuild) {
    Write-Header "Rebuilding Images (--no-cache)"
    Write-Step "Building consumer image..."
    docker compose build --no-cache consumer
    Write-Step "Building producer image..."
    docker compose build --no-cache producer
    Write-Ok "Images rebuilt."
}

# ---------------------------------------------------------------------------
# Start the pipeline
# ---------------------------------------------------------------------------
Write-Header "Starting All Services"
Write-Step "Running: docker compose up -d"
docker compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] docker compose up failed. Check the output above." -ForegroundColor Red
    exit 1
}
Write-Ok "All containers started in detached mode."

# ---------------------------------------------------------------------------
# Wait for services to initialise
# ---------------------------------------------------------------------------
$WaitSeconds = 30
Write-Header "Waiting $WaitSeconds Seconds for Services to Initialise"

for ($i = 1; $i -le $WaitSeconds; $i++) {
    $remaining = $WaitSeconds - $i
    Write-Progress `
        -Activity "Waiting for Zookeeper → Kafka → Spark → Producer → Consumer" `
        -Status   "$remaining second(s) remaining" `
        -PercentComplete ([int](($i / $WaitSeconds) * 100))
    Start-Sleep -Seconds 1
}
Write-Progress -Activity "Waiting" -Completed
Write-Ok "Wait complete."

# ---------------------------------------------------------------------------
# Show container status
# ---------------------------------------------------------------------------
Write-Header "Container Status"
docker compose ps

# ---------------------------------------------------------------------------
# Open Spark Master UI
# ---------------------------------------------------------------------------
Write-Header "Opening Spark Master UI"
$sparkUrl = "http://localhost:8080"
Write-Step "Opening $sparkUrl in your default browser..."
try {
    Start-Process $sparkUrl
    Write-Ok "Browser opened -> $sparkUrl"
} catch {
    Write-Warn "Could not open browser automatically. Navigate to $sparkUrl manually."
}

# ---------------------------------------------------------------------------
# Live tail — producer logs filtered to [FRAUD] lines only
# ---------------------------------------------------------------------------
Write-Header "Live Fraud Transaction Feed  (Ctrl+C to stop)"
Write-Host "  Streaming [FRAUD] lines from fraud-producer logs..." -ForegroundColor Gray
Write-Host "  Each line represents a transaction flagged as fraud." -ForegroundColor Gray
Write-Host ""

# docker compose logs --follow streams continuously; Select-String filters
# to lines containing [FRAUD] and writes them with a timestamp prefix.
try {
    docker compose logs --follow --no-log-prefix producer 2>&1 | ForEach-Object {
        # Match the producer's actual format: [Producer] Sent [FRAUD] transaction ...
        if ($_ -match "\[FRAUD\]") {
            $ts = Get-Date -Format "HH:mm:ss"
            Write-Host "[$ts]  $_" -ForegroundColor Red
        }
    }
} catch [System.Management.Automation.PipelineStoppedException] {
    # Normal Ctrl+C exit from the pipeline — not an error
} finally {
    Write-Host ""
    Write-Ok "Log tail stopped."
    Write-Host ""
    Write-Host "  Containers are still running in the background." -ForegroundColor Gray
    Write-Host "  Run  .\run_demo.ps1 -Down  to stop everything."  -ForegroundColor Gray
    Write-Host ""
}
