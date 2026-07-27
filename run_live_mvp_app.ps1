$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$scriptPath = Join-Path $root "live_mvp_app.py"
$outputDir = Join-Path $root "outputs"
$datasetDir = Join-Path $root "dataset"
$modelPath = Join-Path $root "runs\detect\runs_tx2\yolo11n_tubos_v1\weights\best.pt"
$cameraIp = "10.14.115.241"

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
    Write-Host "Python was not found. Adjust the executable path in this launcher."
    exit 1
}

if (-not $env:AXIS_USER) {
    $env:AXIS_USER = Read-Host "AXIS username"
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

# This MVP is live by default: Python reads the AXIS camera directly through RTSP.
# If the camera requires auth, set AXIS_USER and AXIS_PASSWORD before running.
# To test with a file temporarily, change --source rtsp to --source video and pass --video.
& $pythonExe $scriptPath `
  --source rtsp `
  --camera-ip $cameraIp `
  --codec h264 `
  --camera-resolution 1920x1080 `
  --output-dir $outputDir `
  --dataset-dir $datasetDir `
  --model $modelPath `
  --conf 0.50 `
  --capture-fps 10 `
  --process-fps 10 `
  --buffer-seconds 2 `
  --buffer-max-frames 60 `
  --record-seconds 8 `
  --record-fps 10 `
  --max-clips 100 `
  --plc-enabled `
  --plc-edge rising `
  --port 8767
