param(
    [int]$Mau = 183,
    [switch]$KhongMoAnh
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'Khong tim thay Python trong .venv. Xem demo/README.md de chuan bi moi truong.'
}

Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m demo.so_sanh_mo_hinh --sample $Mau
    if ($LASTEXITCODE -ne 0) {
        throw 'Du doan chua thanh cong. Xem thong bao loi phia tren.'
    }

    $imagePath = Join-Path $projectRoot "demo\output\sample_$Mau.png"
    if (-not $KhongMoAnh) {
        Invoke-Item -LiteralPath $imagePath
    }
} finally {
    Pop-Location
}
