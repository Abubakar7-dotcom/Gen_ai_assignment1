# setup_gpu_pc.ps1 — one-time setup of the RTX 4050 machine for GenAI Assignment 1
# Run from the cloned repo, in PowerShell AS ADMINISTRATOR:
#   Set-ExecutionPolicy -Scope Process Bypass -Force; .\scripts\setup_gpu_pc.ps1
# Safe to re-run: every step checks before installing.
# Creates .venv inside the repo and writes gpu_setup_report.txt to the Desktop.

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot          # repo root (this file lives in scripts\)
$Log = "$env:USERPROFILE\Desktop\gpu_setup_report.txt"
function Say($m) { Write-Host "`n=== $m" -ForegroundColor Cyan; Add-Content $Log "=== $m" }
function RefreshPath { $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User") }
Set-Content $Log "GPU setup report  $(Get-Date)"
Add-Content $Log "Repo: $ProjectDir"

# 0. Admin check (needed for winget installs and power settings)
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Run this in PowerShell as Administrator." -ForegroundColor Red; exit 1
}

# 1. GPU + driver -> pick the matching PyTorch CUDA wheel
Say "GPU / driver"
$smi = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null
if (-not $smi) { Write-Host "nvidia-smi not found. Install the latest NVIDIA Game Ready/Studio driver first." -ForegroundColor Red; exit 1 }
$smi | Tee-Object -FilePath $Log -Append
$driver = [double](($smi -split ",")[2].Trim().Split(".")[0])
if     ($driver -ge 570) { $cu = "cu128" }
elseif ($driver -ge 560) { $cu = "cu126" }
else                     { $cu = "cu118" }
Say "Using PyTorch wheel index: $cu"

# 2. Tools via winget
Say "Installing Python 3.11 and Git (skips if present)"
foreach ($id in @("Python.Python.3.11","Git.Git")) {
    # --source winget: without it a broken msstore source makes winget refuse to install
    $installed = winget list --id $id -e --source winget 2>$null | Select-String $id
    if (-not $installed) { winget install --id $id -e --source winget --silent --accept-package-agreements --accept-source-agreements }
    else { Write-Host "$id already installed" }
}
RefreshPath

# 3. Power: never sleep on AC, lid close does nothing (training runs for hours)
Say "Power settings"
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT

# 4. Project venv + ML stack (torch first with the CUDA index, then the rest from requirements.txt)
Say "Python venv at $ProjectDir\.venv"
Set-Location $ProjectDir
if (-not (Test-Path ".venv")) { py -3.11 -m venv .venv }
$py = "$ProjectDir\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$cu"
& $py -m pip install -r requirements.txt

# 5. Verify CUDA end-to-end
Say "CUDA check"
$check = @'
import torch, time
ok = torch.cuda.is_available()
print("torch", torch.__version__, "| cuda available:", ok)
if ok:
    print("device:", torch.cuda.get_device_name(0), "| VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9,1))
    x = torch.randn(4096, 4096, device="cuda"); torch.cuda.synchronize(); t=time.time()
    for _ in range(20): y = x @ x
    torch.cuda.synchronize(); print("matmul bench s:", round(time.time()-t,3))
'@
$check | & $py - 2>&1 | Tee-Object -FilePath $Log -Append

Say "DONE. Report saved to $Log"
Write-Host @"
Manual steps left:
  1. Settings > Windows Update > Pause updates (pause for at least 1 week).
  2. Keep the laptop plugged in.
  3. Continue with docs\gpu_runbook.md (activate the venv: .venv\Scripts\Activate.ps1).
"@ -ForegroundColor Yellow
