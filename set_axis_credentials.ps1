param(
    [string]$CameraIp = "10.14.115.241",
    [string]$Username = "root"
)

$ErrorActionPreference = "Stop"

$candidates = @(
    (Get-Command python -ErrorAction SilentlyContinue).Source
    (Get-Command python3 -ErrorAction SilentlyContinue).Source
    "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe"
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)

$pythonExe = $candidates |
    Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
    Select-Object -First 1

if (-not $pythonExe) {
    throw "Python was not found."
}

$validationScript = @'
import os
import sys

import requests
from requests.auth import HTTPDigestAuth

url = (
    f"http://{os.environ['TX2_AXIS_CAMERA_IP']}"
    "/axis-cgi/jpg/image.cgi?resolution=320x180"
)
response = requests.get(
    url,
    auth=HTTPDigestAuth(
        os.environ["AXIS_USER"],
        os.environ["AXIS_PASSWORD"],
    ),
    timeout=10,
    stream=True,
)
print(f"AXIS response: HTTP {response.status_code}")
raise SystemExit(0 if response.status_code == 200 else 1)
'@

while ($true) {
    $securePassword = Read-Host "AXIS password for $Username" -AsSecureString
    $passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
    try {
        $env:AXIS_USER = $Username
        $env:AXIS_PASSWORD = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
        $env:TX2_AXIS_CAMERA_IP = $CameraIp
        $validationScript | & $pythonExe -
        $valid = $LASTEXITCODE -eq 0
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
    }

    if ($valid) {
        break
    }

    Remove-Item Env:AXIS_PASSWORD -ErrorAction SilentlyContinue
    Write-Host "AXIS rejected the password. Try again." -ForegroundColor Red
}

[Environment]::SetEnvironmentVariable("AXIS_USER", $env:AXIS_USER, "User")
[Environment]::SetEnvironmentVariable("AXIS_PASSWORD", $env:AXIS_PASSWORD, "User")
Remove-Item Env:TX2_AXIS_CAMERA_IP -ErrorAction SilentlyContinue

Write-Host "AXIS credentials validated and saved for the current Windows user." -ForegroundColor Green
