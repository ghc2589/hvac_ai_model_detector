#!/usr/bin/env bash
set -euo pipefail

# === Carpetas ================================================================
WORKDIR="$(pwd)/paddleocr_onnx"   # carpeta temporal
MODELDIR="$(pwd)/../AI-Models"    # carpeta final para los .onnx
mkdir -p "$WORKDIR" "$MODELDIR"
cd "$WORKDIR"

# === Dependencias ============================================================
echo "=== Instalando dependencias (PaddlePaddle CPU) ==="
python -m pip install --upgrade pip
pip install --no-deps --force-reinstall paddlepaddle paddleocr paddle2onnx onnxruntime

# === Enlaces de modelo =======================================================
# Detector (PP-OCRv3, idioma-agnóstico)
DETMODEL_URL="https://huggingface.co/WSMD/PaddleOCR-ModelBackup/resolve/main/en_PP-OCRv3_det_infer.tar"
# Reconocedor (PP-OCRv4, inglés)
RECMODEL_URL="https://huggingface.co/WSMD/PaddleOCR-ModelBackup/resolve/main/en_PP-OCRv4_rec_infer.tar"

echo "=== Descargando modelos de PaddleOCR ==="
echo "→ Detector (v3)…"
wget --show-progress "$DETMODEL_URL" -O det.tar
echo "→ Reconocedor (v4)…"
wget --show-progress "$RECMODEL_URL" -O rec.tar

echo "=== Descomprimiendo ==="
tar -xf det.tar && rm det.tar
tar -xf rec.tar && rm rec.tar

# === Conversión a ONNX =======================================================
echo "=== Convirtiendo a ONNX ==="
paddle2onnx \
  --model_dir        ./en_PP-OCRv3_det_infer \
  --model_filename   inference.pdmodel \
  --params_filename  inference.pdiparams \
  --save_file        "$MODELDIR/det_ppocrv3_en.onnx" \
  --enable_onnx_checker True

paddle2onnx \
  --model_dir        ./en_PP-OCRv4_rec_infer \
  --model_filename   inference.pdmodel \
  --params_filename  inference.pdiparams \
  --save_file        "$MODELDIR/rec_ppocrv4_en.onnx" \
  --enable_onnx_checker True

echo "=== Conversión terminada ==="
echo "Modelos ONNX guardados en: $MODELDIR"
