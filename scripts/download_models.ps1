# Optional: the backend container downloads missing models by itself on first start.
# Windows: powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"                                   # the progress bar makes PS 5.1 downloads very slow
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12   # GitHub needs TLS 1.2 (PS 5.1 default may be older)
$BaseUrl = if ($env:MODEL_BASE_URL) { $env:MODEL_BASE_URL } else { "https://github.com/Abubakar7-dotcom/Gen_ai_assignment1/releases/download/models-v1" }
$Models = "t1_universal","t2_classifier","t2_spec_salt_pepper","t2_spec_blur","t2_spec_occlusion","t3_moe","t4_generator"
New-Item -ItemType Directory -Force -Path models_onnx | Out-Null
foreach ($m in $Models) {
  $out = "models_onnx\$m.onnx"
  if ((Test-Path $out) -and ((Get-Item $out).Length -gt 0)) { Write-Host "ok   $m.onnx (exists)"; continue }
  Write-Host "get  $m.onnx"
  Invoke-WebRequest -Uri "$BaseUrl/$m.onnx" -OutFile "$out.part" -UseBasicParsing
  Move-Item -Force "$out.part" $out                                        # rename only after a complete download
}
Write-Host "All models in .\models_onnx"
