# GPU PC: run both chains unattended, one after the other (Task Scheduler or a detached window).
#   powershell -ExecutionPolicy Bypass -File scripts\gpu_run_all.ps1
# Uses the repo venv without needing it activated. Both chains are resumable, so running this again after
# a stop continues where they left off (finished searches and trainings are skipped in seconds).
# Waits for AC power before starting: on battery the GPU is throttled and the laptop runs flat mid-training.
# Logs (UTF-16): checkpoints\logs\{gan,restoration}_chain.log, run_all.log; done.flag once both chains ended.
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:Path = "$Root\.venv\Scripts;" + $env:Path
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$Logs = "$Root\checkpoints\logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
Remove-Item "$Logs\done.flag" -ErrorAction SilentlyContinue
function Note($m) { "$(Get-Date -Format s)  $m" | Add-Content "$Logs\run_all.log" }

Note "started"
while ((Get-CimInstance Win32_Battery).BatteryStatus -eq 1) {      # 1 = discharging (no charger); no battery -> $null
    Note "on battery, waiting for AC power"
    Start-Sleep -Seconds 60
}
foreach ($chain in "gan", "restoration") {                         # GAN first: longest single training
    $log = "$Logs\${chain}_chain.log"
    if (Test-Path $log) { Move-Item $log "$Logs\${chain}_chain.prev.log" -Force }   # keep the previous run's log
    Note "chain $chain started"
    powershell -NoProfile -ExecutionPolicy Bypass -File "$Root\scripts\gpu_chain_$chain.ps1" *> $log
    Note "chain $chain ended with exit code $LASTEXITCODE"
}
"finished $(Get-Date -Format s)" | Set-Content "$Logs\done.flag"
Note "all chains ended"
