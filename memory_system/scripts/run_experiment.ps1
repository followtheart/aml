$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project Python environment not found: $python"
}

Push-Location -LiteralPath $projectRoot
try {
    foreach ($directory in @('logs', 'data\results')) {
        $path = Join-Path $projectRoot $directory
        if (Test-Path -LiteralPath $path -PathType Container) {
            Get-ChildItem -LiteralPath $path -File | ForEach-Object {
                Remove-Item -LiteralPath $_.FullName
            }
        }
    }

    # local_eval.py creates its own temporary database for each run.
    & $python scripts/local_eval.py --data .\data\prepared\personamem-v2-32k.jsonl --convs 1 --limit 30 --output data/results/personamem-v2-32k.jsonl
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
