#!/usr/bin/env bash
# Optional: the backend container downloads missing models by itself on first start.
# Usage: bash scripts/download_models.sh   (fills ./models_onnx)
set -euo pipefail
BASE_URL="${MODEL_BASE_URL:-https://github.com/Abubakar7-dotcom/Gen_ai_assignment1/releases/download/models-v1}"
MODELS="t1_universal t2_classifier t2_spec_salt_pepper t2_spec_blur t2_spec_occlusion t3_moe t4_generator"
mkdir -p models_onnx
for m in $MODELS; do
  if [ -s "models_onnx/$m.onnx" ]; then echo "ok   $m.onnx (exists)"; continue; fi
  echo "get  $m.onnx"
  curl -fL --retry 3 -o "models_onnx/$m.onnx.part" "$BASE_URL/$m.onnx"
  mv "models_onnx/$m.onnx.part" "models_onnx/$m.onnx"     # rename only after a complete download
done
echo "All models in ./models_onnx"
