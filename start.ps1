#requires -Version 5.1
<#
.SYNOPSIS
    Startet Kivski Tactical AI Simulator komplett (Backend + Frontend + Browser).

.DESCRIPTION
    Oeffnet zwei Hintergrund-Prozesse:
      1. FastAPI + WebSocket Backend (kivski-serve auf Port 8000)
      2. Vite Dev-Server fuers React Frontend (Port 5173)
    Wartet kurz bis beide bereit sind und oeffnet dann den Browser
    auf http://localhost:5173.

.PARAMETER NoBrowser
    Browser nicht automatisch oeffnen.

.PARAMETER BackendPort
    Backend Port (default 8000).

.PARAMETER FrontendPort
    Vite Port (default 5173).

.EXAMPLE
    .\start.ps1
    .\start.ps1 -NoBrowser
    .\start.ps1 -BackendPort 8765
#>
param(
    [switch]$NoBrowser,
    [int]$BackendPort = 8000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  Kivski Tactical AI Simulator - Launcher" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""

# --- Pruefe venv -------------------------------------------------------------
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "[!] Kein .venv gefunden. Erstelle eines mit:" -ForegroundColor Yellow
    Write-Host "    py -3.14 -m venv .venv" -ForegroundColor Yellow
    Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
    Write-Host "    pip install -e `".[dev]`"" -ForegroundColor Yellow
    exit 1
}

# --- Pruefe node_modules -----------------------------------------------------
$nodeModules = Join-Path $root "node_modules"
$webNodeModules = Join-Path $root "apps\web\node_modules"
if (-not (Test-Path $nodeModules) -or -not (Test-Path $webNodeModules)) {
    Write-Host "[i] node_modules fehlt. Installiere via 'npm install' ..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[X] npm install fehlgeschlagen." -ForegroundColor Red
        exit 1
    }
}

# --- Backend starten (Hintergrund) ------------------------------------------
$backendLog = Join-Path $root "models\logs\backend.log"
New-Item -ItemType File -Force -Path $backendLog | Out-Null

$serveScript = Join-Path $root "scripts\serve.py"
Write-Host "[*] Backend (FastAPI + WebSocket) -> Port $BackendPort" -ForegroundColor Green
$backendProc = Start-Process -FilePath $venvPython `
    -ArgumentList @($serveScript, "--port", "$BackendPort") `
    -WorkingDirectory $root `
    -RedirectStandardOutput $backendLog `
    -RedirectStandardError "$backendLog.err" `
    -WindowStyle Hidden `
    -PassThru

Write-Host "    PID: $($backendProc.Id)  log: models\logs\backend.log" -ForegroundColor DarkGray

# --- Auf Backend warten ------------------------------------------------------
Write-Host "[*] Warte auf Backend-Bereitschaft ..." -ForegroundColor Green
$healthUrl = "http://127.0.0.1:$BackendPort/api/health"
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    Write-Host "[X] Backend nicht erreichbar nach 15s. Siehe models\logs\backend.log" -ForegroundColor Red
    Stop-Process -Id $backendProc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "    Backend OK." -ForegroundColor DarkGreen

# --- Frontend starten (Hintergrund) -----------------------------------------
$frontendLog = Join-Path $root "models\logs\frontend.log"
New-Item -ItemType File -Force -Path $frontendLog | Out-Null

Write-Host "[*] Frontend (Vite Dev-Server) -> Port $FrontendPort" -ForegroundColor Green
$env:VITE_API_PROXY_TARGET = "http://127.0.0.1:$BackendPort"
$frontendProc = Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--", "--port", "$FrontendPort") `
    -WorkingDirectory $root `
    -RedirectStandardOutput $frontendLog `
    -RedirectStandardError "$frontendLog.err" `
    -WindowStyle Hidden `
    -PassThru

Write-Host "    PID: $($frontendProc.Id)  log: models\logs\frontend.log" -ForegroundColor DarkGray

# --- Auf Frontend warten -----------------------------------------------------
Write-Host "[*] Warte auf Vite-Bereitschaft ..." -ForegroundColor Green
$frontendUrl = "http://localhost:$FrontendPort"
$ready = $false
for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $r = Invoke-WebRequest -Uri $frontendUrl -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
}
if (-not $ready) {
    Write-Host "[!] Vite-Server brauchte zu lang, versuche trotzdem zu oeffnen." -ForegroundColor Yellow
} else {
    Write-Host "    Frontend OK." -ForegroundColor DarkGreen
}

# --- Browser oeffnen ---------------------------------------------------------
if (-not $NoBrowser) {
    Write-Host "[*] Oeffne Browser: $frontendUrl" -ForegroundColor Green
    Start-Process $frontendUrl
}

# --- PID-Datei schreiben -----------------------------------------------------
$pidFile = Join-Path $root ".kivski-pids.json"
@{
    backend_pid  = $backendProc.Id
    frontend_pid = $frontendProc.Id
    started_at   = (Get-Date).ToString("o")
    backend_url  = "http://127.0.0.1:$BackendPort"
    frontend_url = $frontendUrl
} | ConvertTo-Json | Out-File -FilePath $pidFile -Encoding utf8

Write-Host ""
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  Kivski laeuft!" -ForegroundColor Cyan
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host "  Frontend : $frontendUrl" -ForegroundColor White
Write-Host "  Backend  : http://127.0.0.1:$BackendPort/api/health" -ForegroundColor White
Write-Host "  Logs     : models\logs\backend.log / frontend.log" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Zum Beenden:  .\stop.ps1" -ForegroundColor Yellow
Write-Host "===========================================" -ForegroundColor Cyan
Write-Host ""
