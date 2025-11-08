# PowerShell script to run the example prediction script on Windows
# Usage: .\run_example.ps1

Write-Host "=" -NoNewline
Write-Host ("=" * 59)
Write-Host "WILDFIRE IMPACT PREDICTION - EXAMPLE USAGE"
Write-Host ("=" * 60)
Write-Host ""

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Run the example script
Write-Host "Running example prediction script..."
python src/example_prediction.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Example completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Example failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

