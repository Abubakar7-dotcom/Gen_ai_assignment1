# GPU PC, terminal A: Tasks 1-3 end to end (tune -> train -> evaluate). Resumable: just re-run after a crash.
# Run from repo root with the venv active:  powershell -ExecutionPolicy Bypass -File scripts\gpu_chain_restoration.ps1
$ErrorActionPreference = "Stop"
function Step($cmd) { Write-Host "`n>>> $cmd" -ForegroundColor Cyan; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "failed: $cmd" } }

Step "python -m optuna_studies.tune --task t1"
Step "python -m train.train --task t1"
Step "python -m optuna_studies.tune --task t2_cls"
Step "python -m train.train --task t2_cls"
Step "python -m optuna_studies.tune --task t2_spec"
foreach ($s in "salt_pepper","blur","occlusion") { Step "python -m train.train --task t2_spec --specialist $s" }
Step "python -m optuna_studies.tune --task t3"
Step "python -m train.train --task t3"
foreach ($t in "t1","t2","t3") { Step "python -m eval.evaluate --task $t" }
Write-Host "`nRestoration chain finished. Commit results/, configs/*_best.yaml, optuna_studies/*.db" -ForegroundColor Green
