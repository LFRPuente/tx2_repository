$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$frontendDir = Join-Path $root "mvp_react_app"
$backendScript = Join-Path $root "run_homography_web_app.ps1"

$backendOk = $false
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:5050/api/meta" -UseBasicParsing -TimeoutSec 2 | Out-Null
    $backendOk = $true
} catch {
    $backendOk = $false
}

if (-not $backendOk) {
    Start-Process powershell -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-File",$backendScript -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

if (-not (Test-Path (Join-Path $frontendDir "node_modules"))) {
    Push-Location $frontendDir
    cmd /c npm install
    Pop-Location
}

Push-Location $frontendDir
cmd /c npm run dev
Pop-Location
