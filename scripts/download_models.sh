#!/usr/bin/env bash
# Downloads the trained ONNX models into ./models_onnx (used by docker compose).
# Usage: bash scripts/download_models.sh
set -euo pipefail
BASE_URL="${MODEL_BASE_URL:-https://github.com/Abubakar7-dotcom/Gen_ai_assignment1/releases/download/models-v1}"
MODELS="t1_universal t2_classifier t2_spec_salt_pepper t2_spec_blur t2_spec_occlusion t3_moe t4_generator"
mkdir -p models_onnx
for m in $MODELS; do
  if [ -s "models_onnx/$m.onnx" ]; then echo "ok   $m.onnx (exists)"; continue; fi
  echo "get  $m.onnx"
  curl -fL --retry 3 -o "models_onnx/$m.onnx" "$BASE_URL/$m.onnx"
done
echo "All models in ./models_onnx"
