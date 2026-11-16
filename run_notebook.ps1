$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project virtual environment not found. Create .venv and install requirements first."
}

& $python -m jupyter lab (Join-Path $PSScriptRoot "COMP5423_SearchAgent.ipynb") --ServerApp.open_browser=True
