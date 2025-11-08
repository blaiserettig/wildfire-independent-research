# PowerShell script to run the training script on Windows
# Usage: .\run_training.ps1

Write-Host "=" -NoNewline
Write-Host ("=" * 59)
Write-Host "WILDFIRE IMPACT PREDICTION MODEL - TRAINING"
Write-Host ("=" * 60)
Write-Host ""

# Change to script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Run the training script
Write-Host "Running training script..."
python src/train_and_evaluate.py

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Training completed successfully!" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

