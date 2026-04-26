Write-Host "[1/3] Checking Python environment"
Write-Host "Estimated time: 2-10 seconds"

python --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "Python is not available on PATH."
    exit 1
}

Write-Host "[2/3] Starting demo train"
Write-Host "Estimated time: 10-40 seconds on CPU"

python train.py `
  --data-root demo_data/PGDP5K_demo `
  --ext-root demo_data/PGDP5K_demo/Ext-PGDP5K `
  --epochs 1 `
  --batch-size 2 `
  --device cpu `
  --output-dir outputs/demo_run `
  --seed 42

if ($LASTEXITCODE -ne 0) {
    Write-Error "Demo run failed."
    exit 1
}

Write-Host "[3/3] Done"
Write-Host "Output directory: outputs/demo_run"
