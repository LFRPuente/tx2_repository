param(
    [switch]$Production,
    [switch]$DatabaseDisabled,
    [string]$SqlitePath,
    [ValidateRange(0.01, 1.0)]
    [double]$Confidence = 0.10,
    [ValidateSet("auto", "cpu", "cuda", "cuda:0")]
    [string]$Device = "auto",
    [string]$CameraIp = "10.14.115.241",
    [string]$ListenAddress
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $root $(
    if ($Production) { "run_live_mvp_production.py" } else { "live_mvp_app.py" }
)
$outputDir = Join-Path $root "outputs"
$defaultSqlitePath = Join-Path $outputDir "tx2_live_mvp.sqlite3"
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
    Write-Warning "Individual-piece model not found. Live inference is using the legacy package model."
}
else {
    if ($modelPath -eq $previousPieceModelPath) {
        Write-Warning "Current individual-piece model not found. Live inference is using the previous v2 model."
    }
    elseif ($modelPath -eq $olderPieceModelPath) {
        Write-Warning "Current individual-piece models not found. Live inference is using the older v1 model."
    }
    $modelHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $modelPath).Hash
    Write-Host "Model: individual pieces ($modelPath)"
    Write-Host "Model SHA-256: $modelHash"
}
Write-Host "Camera: $CameraIp"
Write-Host "YOLO confidence: $Confidence"
Write-Host "YOLO device: $Device"
Write-Host "Homography: $(Join-Path $outputDir 'homography_selection.json')"
Write-Host "Calibration: $(Join-Path $outputDir 'table_measurement_calibration.json')"

$candidates = @(
    (Join-Path $root ".venv-gpu\Scripts\python.exe")
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
    Write-Host "Python was not found. Adjust the executable path in this launcher."
    exit 1
}

if (-not $env:AXIS_USER) {
    $env:AXIS_USER = [Environment]::GetEnvironmentVariable("AXIS_USER", "User")
}

if (-not $env:AXIS_USER) {
    $env:AXIS_USER = Read-Host "AXIS username"
}

if (-not $env:AXIS_PASSWORD) {
    $env:AXIS_PASSWORD = [Environment]::GetEnvironmentVariable("AXIS_PASSWORD", "User")
}

if (-not $env:AXIS_PASSWORD) {
    $securePassword = Read-Host "AXIS password" -AsSecureString
    $passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
    try {
        $env:AXIS_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
    }
}

$databaseArgs = @()
if ($DatabaseDisabled) {
    $databaseArgs += "--db-disabled"
    Write-Warning "The database is disabled explicitly. History persistence is in simulation mode."
}
elseif ($env:TX2_POSTGRES_DSN) {
    Write-Host "Database: PostgreSQL"
}
else {
    if (-not $SqlitePath) {
        $SqlitePath = if ($env:TX2_SQLITE_PATH) { $env:TX2_SQLITE_PATH } else { $defaultSqlitePath }
    }
    $databaseArgs += @("--sqlite-path", $SqlitePath)
    Write-Host "Database: SQLite temporal ($SqlitePath)"
}

$serverArgs = @()
if ($Production) {
    if (-not $ListenAddress) {
        $ListenAddress = if ($env:TX2_LISTEN_ADDRESS) {
            $env:TX2_LISTEN_ADDRESS
        }
        else {
            "127.0.0.1"
        }
    }
    $serverArgs += @(
        "--waitress-host", $ListenAddress,
        "--waitress-threads", "16"
    )
    Write-Host "Server: Waitress on ${ListenAddress}:8767"
}

# This MVP is live by default: Python reads the AXIS camera directly through RTSP.
# If the camera requires auth, set AXIS_USER and AXIS_PASSWORD before running.
# To test with a file temporarily, change --source rtsp to --source video and pass --video.
& $pythonExe $scriptPath `
  --source rtsp `
  --camera-ip $CameraIp `
  --codec h264 `
  --camera-resolution 2880x2160 `
  --camera-decoder nvdec `
  --output-dir $outputDir `
  --dataset-dir $datasetDir `
  --model $modelPath `
  --device $Device `
  --conf $Confidence `
  --capture-fps 30 `
  --live-stream-fps 30 `
  --buffer-seconds 3 `
  --buffer-max-frames 60 `
  --record-seconds 8 `
  --record-fps 10 `
  --video-encoder nvenc `
  --video-bitrate-mbps 16 `
  --processing-cache-frames 320 `
  --pre-trigger-seconds 2 `
  --save-raw-clips `
  --raw-camera-resolution 2880x2160 `
  --raw-record-fps 10 `
  --measurement-delay-seconds 0 `
  --max-clips 100 `
  --plc-enabled `
  --plc-edge rising `
  --port 8767 `
  @databaseArgs `
  @serverArgs
