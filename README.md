# Quantized LLM + RAG (FastAPI + FAISS + Phi‑3)

## Goal
Deploy a small, low‑cost LLM with 4‑bit quantization + RAG, exposed via a clean FastAPI service that can run on CPU‑only servers (e.g., Azure Container Instances).

FastAPI API serving a 4‑bit GGUF LLM with a lightweight FAISS RAG pipeline. Designed for low‑cost CPU servers (Azure Container Instances) and local Mac testing.

## Features
- 4‑bit quantized Phi‑3 GGUF (llama.cpp via `llama-cpp-python`)
- Simple RAG with FAISS (cosine similarity)
- Wikipedia public-source ingestion (replaceable)
- Docker image ready for ACI

## Repo structure
```
app/
  main.py        # FastAPI app
  rag.py         # FAISS utilities
  ingest.py      # build index from public sources
  settings.py    # config via env
scripts/
  download_model.py
Dockerfile
requirements.txt
```

## Local dev (Mac)

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# Download 4-bit Phi-3 GGUF (also uploaded on my HF: [sitsope/phi-3-mini-4k-instruct-q4](https://huggingface.co/sitsope/phi-3-mini-4k-instruct-q4))
python scripts/download_model.py \
  --repo microsoft/Phi-3-mini-4k-instruct-gguf \
  --filename Phi-3-mini-4k-instruct-q4.gguf \
  --out models

# Build FAISS index from public pages
python -m app.ingest --pages "Large_language_model,Azure,Quantization_(signal_processing)" --lang en

# Run API
export MODEL_PATH="models/Phi-3-mini-4k-instruct-q4.gguf"
export N_GPU_LAYERS="-1"   # Metal offload on Mac
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Test:
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What is quantization in signal processing?"}'
```

## Docker (local)

Build:
```bash
docker build -t quant-llm .
```

Run:
```bash
docker run --rm -p 8000:8000 \
  -e MODEL_PATH=/models/Phi-3-mini-4k-instruct-q4.gguf \
  -v "$PWD/models:/models" \
  quant-llm
```

## Azure Container Instances (ACI)

1) Build + push to ACR:
```bash
az group create -n rg-quant-llm -l westeurope
az acr create -n acrquantllm -g rg-quant-llm --sku Basic
az acr login -n acrquantllm
az acr build -t quant-llm:1 -r acrquantllm .
```

2) Run in ACI (downloads model at startup):
```bash
az container create \
  -g rg-quant-llm \
  -n quant-llm-api \
  --image acrquantllm.azurecr.io/quant-llm:1 \
  --registry-login-server acrquantllm.azurecr.io \
  --registry-username <ACR_USERNAME> \
  --registry-password <ACR_PASSWORD> \
  --cpu 2 --memory 6 \
  --ports 8000 \
  --environment-variables MODEL_PATH=/models/Phi-3-mini-4k-instruct-q4.gguf N_THREADS=2 N_GPU_LAYERS=0 \
  --command-line "bash -lc 'python scripts/download_model.py --repo microsoft/Phi-3-mini-4k-instruct-gguf --filename Phi-3-mini-4k-instruct-q4.gguf --out /models && uvicorn app.main:app --host 0.0.0.0 --port 8000'"
```

3) Get public IP:
```bash
az container show -g rg-quant-llm -n quant-llm-api --query ipAddress.ip -o tsv
```

## Config
Environment variables in `app/settings.py`:
- `MODEL_PATH` (default: `models/phi-3-mini-4k-instruct-q4.gguf`)
- `N_CTX` (default: 4096)
- `N_THREADS` (default: 8)
- `N_GPU_LAYERS` (default: 0, use `-1` on Mac for Metal)
- `RAG_TOP_K` (default: 4)

## Notes
- 4‑bit GGUF is the best CPU-friendly option for cost/memory.
- RAG sources are currently Wikipedia; swap `app/ingest.py` to your own docs.

## Contributing
See `CONTRIBUTING.md`.

## License
MIT. See `LICENSE`.
