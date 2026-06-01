$scriptPath = "C:\Users\luis_\OneDrive\Desktop\tx2_cv\warp_player_app.py"

$candidates = @(
    (Get-Command python -ErrorAction SilentlyContinue).Source
    (Get-Command python3 -ErrorAction SilentlyContinue).Source
    (Get-Command py -ErrorAction SilentlyContinue).Source
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    "$env:ProgramFiles\Python311\python.exe"
    "$env:ProgramFiles\Python312\python.exe"
)

$pythonExe = $null
foreach ($candidate in $candidates) {
    if (-not $candidate) { continue }
    if (Test-Path -LiteralPath $candidate) {
        $item = Get-Item -LiteralPath $candidate
        if ($item.Length -gt 0) {
            $pythonExe = $candidate
            break
        }
    }
}

if (-not $pythonExe) {
    Write-Host "No se encontro un ejecutable real de Python en esta maquina."
    Write-Host "Abre el script manualmente con tu Python instalado o ajusta la ruta en este lanzador."
    exit 1
}

& $pythonExe $scriptPath
