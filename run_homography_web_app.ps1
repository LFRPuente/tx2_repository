$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $root "homography_web_app.py"
$outputDir = Join-Path $root "outputs"
$videoDir = Join-Path $outputDir "live_plc_clips"
$datasetDir = Join-Path $root "dataset_pieces"
$pieceModelPath = Join-Path $root "runs\detect\runs_tx2\yolo11n_pieces_v3\weights\best.pt"
$previousPieceModelPath = Join-Path $root "runs\detect\runs_tx2\yolo11n_pieces_v2\weights\best.pt"
$olderPieceModelPath = Join-Path $root "runs\detect\runs_tx2\yolo11n_pieces_v1\weights\best.pt"
$legacyModelPath = Join-Path $root "runs\detect\runs_tx2\yolo11n_tubos_v1\weights\best.pt"
$modelPath = @($pieceModelPath, $previousPieceModelPath, $olderPieceModelPath, $legacyModelPath) |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1

if (-not $modelPath) {
    throw "No YOLO model was found. Expected the current model at: $pieceModelPath"
}
if ($modelPath -eq $legacyModelPath) {
    Write-Warning "Individual-piece model not found. Annotation uses dataset_pieces; prediction still uses the legacy package model."
}
elseif ($modelPath -eq $previousPieceModelPath) {
    Write-Warning "Current individual-piece model not found. Annotation is using the previous v2 model."
}
elseif ($modelPath -eq $olderPieceModelPath) {
    Write-Warning "Current individual-piece models not found. Annotation is using the older v1 model."
}

$modelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelPath).Hash
Write-Host "Model: $modelPath"
Write-Host "Model SHA-256: $modelHash"

$candidates = @(
    (Get-Command python -ErrorAction SilentlyContinue).Source
    (Get-Command python3 -ErrorAction SilentlyContinue).Source
    (Get-Command py -ErrorAction SilentlyContinue).Source
    "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe"
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

if (-not (Get-ChildItem -LiteralPath $videoDir -Recurse -File -Filter "*_raw.mp4" -ErrorAction SilentlyContinue | Select-Object -First 1)) {
    Write-Host "No hay clips raw del Live MVP todavia. Espera un evento PLC valido y vuelve a iniciar la herramienta."
    exit 1
}

& $pythonExe $scriptPath `
  --video-dir $videoDir `
  --latest-video-format-only `
  --second 30.0 `
  --output-dir $outputDir `
  --dataset-dir $datasetDir `
  --model $modelPath `
  --port 5050
