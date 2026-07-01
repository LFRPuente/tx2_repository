$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $root "live_mvp_app.py"
$videoPath = "C:\Users\luis_\Downloads\20260508_000307_7F66.mkv"
$outputDir = Join-Path $root "outputs"
$datasetDir = Join-Path $root "dataset"
$modelPath = Join-Path $root "runs\detect\runs_tx2\yolo11n_tubos_v1\weights\best.pt"

$candidates = @(
    (Get-Command python -ErrorAction SilentlyContinue).Source
    (Get-Command python3 -ErrorAction SilentlyContinue).Source
    (Get-Command py -ErrorAction SilentlyContinue).Source
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    "$env:ProgramFiles\Python312\python.exe"
    "$env:ProgramFiles\Python311\python.exe"
)

$pythonExe = $null
foreach ($candidate in $candidates) {
    if (-not $candidate) { continue }
    if (Test-Path -LiteralPath $candidate) {
        $pythonExe = $candidate
        break
    }
}

if (-not $pythonExe) {
    Write-Host "No se encontro Python. Ajusta la ruta del ejecutable en este lanzador."
    exit 1
}

# Default source is video, so the MVP can be tested without camera credentials.
# For RTSP, change --source video to --source rtsp and set AXIS_USER / AXIS_PASSWORD.
# For PLC-triggered clips, add --plc-enabled.
& $pythonExe $scriptPath `
  --source video `
  --video $videoPath `
  --output-dir $outputDir `
  --dataset-dir $datasetDir `
  --model $modelPath `
  --conf 0.50 `
  --capture-fps 15 `
  --process-fps 2 `
  --record-seconds 10 `
  --port 8767
