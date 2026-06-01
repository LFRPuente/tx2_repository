$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $root "homography_web_app.py"
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

& $pythonExe $scriptPath `
  --video $videoPath `
  --second 155.0 `
  --output-dir $outputDir `
  --dataset-dir $datasetDir `
  --model $modelPath `
  --port 5050
