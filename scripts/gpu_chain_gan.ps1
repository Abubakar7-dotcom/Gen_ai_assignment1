# GPU PC, terminal B (start FIRST, it is the longest): Task 4 tune -> train -> evaluate.
$ErrorActionPreference = "Stop"
function Step($cmd) { Write-Host "`n>>> $cmd" -ForegroundColor Cyan; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "failed: $cmd" } }
Step "python -m optuna_studies.tune --task t4"
Step "python -m train.train --task t4"
Step "python -m eval.evaluate --task t4"
Write-Host "`nGAN chain finished." -ForegroundColor Green
