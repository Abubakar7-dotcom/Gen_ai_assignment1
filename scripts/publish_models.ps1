# Publishes the trained models as the PUBLIC GitHub Release `models-v1` (run once, by the repo owner, on the GPU PC).
#   powershell -ExecutionPolicy Bypass -File scripts\publish_models.ps1
# Uploads the 7 ONNX files, the submitted PyTorch checkpoints (renamed ckpt_*.pt) and SHA256SUMS.txt (~694 MB).
$ErrorActionPreference = "Stop"
$Repo = "Abubakar7-dotcom/Gen_ai_assignment1"
$Stage = Join-Path $env:TEMP "genai_a1_release"
New-Item -ItemType Directory -Force $Stage | Out-Null
Get-ChildItem $Stage -File | Remove-Item

$Checkpoints = [ordered]@{
  "checkpoints\t1\best.pt"                  = "ckpt_t1_best.pt"
  "checkpoints\t2_cls\best.pt"              = "ckpt_t2_cls_best.pt"
  "checkpoints\t2_spec_salt_pepper\best.pt" = "ckpt_t2_spec_salt_pepper_best.pt"
  "checkpoints\t2_spec_blur\best.pt"        = "ckpt_t2_spec_blur_best.pt"
  "checkpoints\t2_spec_occlusion\best.pt"   = "ckpt_t2_spec_occlusion_best.pt"
  "checkpoints\t3\best.pt"                  = "ckpt_t3_best.pt"
  "checkpoints\t3\last.pt"                  = "ckpt_t3_last.pt"
  "checkpoints\t4\best_G.pt"                = "ckpt_t4_best_G.pt"
}
foreach ($src in $Checkpoints.Keys) { Copy-Item $src (Join-Path $Stage $Checkpoints[$src]) }
Copy-Item models_onnx\*.onnx $Stage
$Files = Get-ChildItem $Stage -File | Sort-Object Name
$Files | ForEach-Object { "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.Name } |
  Out-File -Encoding ascii (Join-Path $Stage "SHA256SUMS.txt")

$Notes = @"
Trained models for GenAI Assignment 1 (Oxford-IIIT Pet restoration, FS2K sketch-to-photo).

**ONNX models** used by the app. ``scripts/download_models.sh`` / ``.ps1`` fetch them into ``models_onnx/``.
PyTorch vs ONNX Runtime parity was checked for all 7 (max abs diff 6.6e-6, ``results/onnx_parity.csv``).

**PyTorch checkpoints** (``ckpt_*.pt``) are only needed to re-evaluate, re-export or fine-tune. Put them back as
``checkpoints/<task>/best.pt`` (T4: ``checkpoints/t4/best_G.pt``). ``ckpt_t3_best.pt`` is the submitted T3 model
(best validation, end of warm-up); ``ckpt_t3_last.pt`` is the end of joint fine-tuning (see ``docs/decisions.md``).

``SHA256SUMS.txt`` lists the checksums.
"@
$Assets = (Get-ChildItem $Stage -File | Sort-Object Name).FullName
gh release create models-v1 --repo $Repo --target main --title "Trained models v1" --notes $Notes @Assets
Write-Host "Release: https://github.com/$Repo/releases/tag/models-v1"
