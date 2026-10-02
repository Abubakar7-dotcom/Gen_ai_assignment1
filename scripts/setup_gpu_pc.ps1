# setup_gpu_pc.ps1 — one-time setup of the RTX 4050 machine for GenAI Assignment 1
# Run in PowerShell AS ADMINISTRATOR:
#   Set-ExecutionPolicy -Scope Process Bypass -Force; .\setup_gpu_pc.ps1
# Safe to re-run: every step checks before installing.

$ErrorActionPreference = "Stop"
$ProjectDir = "C:\genai-a1"
$Log = "$env:USERPROFILE\Desktop\gpu_setup_report.txt"
function Say($m) { Write-Host "`n=== $m" -ForegroundColor Cyan; Add-Content $Log "=== $m" }
function RefreshPath { $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [Environment]::GetEnvironmentVariable("Path","User") }
Set-Content $Log "GPU setup report  $(Get-Date)"

# 0. Admin check
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "Run this in PowerShell as Administrator." -ForegroundColor Red; exit 1
}

# 1. GPU + driver
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
Say "Installing Python 3.11, Git, Tailscale (skips if present)"
foreach ($id in @("Python.Python.3.11","Git.Git","Tailscale.Tailscale")) {
    # --source winget: without it a broken msstore source makes winget refuse to install
    $installed = winget list --id $id -e --source winget 2>$null | Select-String $id
    if (-not $installed) { winget install --id $id -e --source winget --silent --accept-package-agreements --accept-source-agreements }
    else { Write-Host "$id already installed" }
}
RefreshPath

# 3. OpenSSH server (for remote work over Tailscale)
Say "OpenSSH server"
$cap = Get-WindowsCapability -Online -Name OpenSSH.Server*
if ($cap.State -ne "Installed") { Add-WindowsCapability -Online -Name $cap.Name | Out-Null }
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
New-ItemProperty -Path "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell `
  -Value "C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe" -PropertyType String -Force | Out-Null
$ak = "C:\ProgramData\ssh\administrators_authorized_keys"
if (-not (Test-Path $ak)) { New-Item $ak -ItemType File | Out-Null }
icacls $ak /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F" | Out-Null
# Hardening: key-only login (no password guessing) and SSH reachable ONLY over Tailscale (100.64.0.0/10)
$cfgPath = "C:\ProgramData\ssh\sshd_config"
$cfg = Get-Content $cfgPath
if (-not ($cfg | Select-String "^PasswordAuthentication no")) {
    Set-Content $cfgPath (@("PasswordAuthentication no") + $cfg)   # prepended so it applies globally, not inside the Match block
}
if (Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue) {
    Set-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -RemoteAddress 100.64.0.0/10
} else {
    New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH (Tailscale only)" -Direction Inbound `
      -Protocol TCP -LocalPort 22 -Action Allow -RemoteAddress 100.64.0.0/10 | Out-Null
}
Restart-Service sshd
Add-Content $Log "SSH user for login: $env:USERNAME   (paste student's public key into $ak)"

# 4. Power: never sleep on AC, lid close does nothing
Say "Power settings"
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0
powercfg /setactive SCHEME_CURRENT

# 5. Project venv + ML stack
Say "Python venv at $ProjectDir\.venv"
New-Item -ItemType Directory -Force -Path $ProjectDir | Out-Null
Set-Location $ProjectDir
if (-not (Test-Path ".venv")) { py -3.11 -m venv .venv }
$py = "$ProjectDir\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$cu"
& $py -m pip install optuna optuna-dashboard wandb pytorch-msssim onnx onnxruntime opencv-python-headless `
    scikit-learn scipy matplotlib seaborn pandas pyyaml tqdm pillow pytest

# 6. Verify CUDA end-to-end
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

# 7. Tailscale info
Say "Tailscale"
RefreshPath
$ts = "C:\Program Files\Tailscale\tailscale.exe"
if (Test-Path $ts) { & $ts status 2>&1 | Tee-Object -FilePath $Log -Append } else { Add-Content $Log "Tailscale not found on PATH yet" }
Add-Content $Log "Hostname: $env:COMPUTERNAME"

Say "DONE. Report saved to $Log"
Write-Host @"
Manual steps left:
  1. Open Tailscale from the Start menu and sign in with the shared Google account.
  2. Settings > Windows Update > Pause updates (pause for at least 1 week).
  3. Keep the laptop plugged in.
"@ -ForegroundColor Yellow
