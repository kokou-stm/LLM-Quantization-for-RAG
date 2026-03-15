#!/bin/sh
set -e

MODEL_PATH=${MODEL_PATH:-/models/Phi-3-mini-4k-instruct-q4.gguf}

if [ ! -f "$MODEL_PATH" ]; then
  python scripts/download_model.py \
    --repo microsoft/Phi-3-mini-4k-instruct-gguf \
    --filename Phi-3-mini-4k-instruct-q4.gguf \
    --out /models
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
