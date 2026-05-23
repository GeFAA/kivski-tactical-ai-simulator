#requires -Version 5.1
<#
.SYNOPSIS
    Stoppt die Kivski-Hintergrundprozesse (Backend + Frontend), die start.ps1 angelegt hat.
#>
param([switch]$Force)

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
Set-Location $root

$pidFile = Join-Path $root ".kivski-pids.json"
if (-not (Test-Path $pidFile)) {
    Write-Host "[i] Keine .kivski-pids.json gefunden. Versuche generischer Cleanup ueber Port-Lookup ..." -ForegroundColor Yellow

    # Generischer Cleanup: alle Python-Prozesse die uvicorn lauschen + alle node die vite lauschen
    $stopped = 0
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in 8000, 5173 } |
        ForEach-Object {
            $procId = $_.OwningProcess
            try {
                Stop-Process -Id $procId -Force -ErrorAction Stop
                Write-Host "  - Stopped PID $procId on port $($_.LocalPort)" -ForegroundColor Green
                $stopped++
            } catch {
                Write-Host "  - Konnte PID $procId nicht stoppen: $($_.Exception.Message)" -ForegroundColor DarkYellow
            }
        }
    if ($stopped -eq 0) {
        Write-Host "[i] Keine laufenden Kivski-Prozesse gefunden." -ForegroundColor DarkGray
    }
    exit 0
}

$data = Get-Content $pidFile -Raw | ConvertFrom-Json

foreach ($name in @("backend_pid", "frontend_pid")) {
    $procId = $data.$name
    if ($procId) {
        try {
            $proc = Get-Process -Id $procId -ErrorAction Stop
            # Auch Child-Processes mit-killen (npm spawnt node)
            Get-CimInstance Win32_Process -Filter "ParentProcessId=$procId" -ErrorAction SilentlyContinue |
                ForEach-Object {
                    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } catch {}
                }
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "[*] Stopped $name (PID $procId)" -ForegroundColor Green
        } catch {
            Write-Host "[i] $name (PID $procId) nicht mehr aktiv." -ForegroundColor DarkGray
        }
    }
}

Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
Write-Host "[OK] Kivski gestoppt." -ForegroundColor Cyan
