$ErrorActionPreference = 'Stop'
$taskWorkspace = Split-Path -Parent $PSScriptRoot
$pointerPath = Join-Path $taskWorkspace '02_データ\v2\長時間計算\実行中.json'
if (-not (Test-Path -LiteralPath $pointerPath)) { Write-Output 'No library job has been registered.'; exit 0 }
$pointer = Get-Content -Raw -LiteralPath $pointerPath -Encoding utf8 | ConvertFrom-Json
$jobFolder = [IO.Path]::GetFullPath($pointer.run_dir)
$allowedRoot = [IO.Path]::GetFullPath((Join-Path $taskWorkspace '02_データ\v2\長時間計算'))
if (-not $jobFolder.StartsWith($allowedRoot + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Unexpected job folder' }
New-Item -ItemType File -Path (Join-Path $jobFolder 'STOP') -Force | Out-Null
Write-Output 'Stop requested. Running cases will finish and results will be saved.'
