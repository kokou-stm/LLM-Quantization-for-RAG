FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build llama-cpp without -march=native to avoid illegal instruction on weaker CPUs
ENV CMAKE_ARGS="-DLLAMA_NATIVE=OFF" \
    FORCE_CMAKE=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data

RUN chmod +x /app/scripts/start.sh

ENV MODEL_PATH="/models/Phi-3-mini-4k-instruct-q4.gguf" \
    N_THREADS="4" \
    N_GPU_LAYERS="0" \
    N_CTX="4096" \
    RAG_TOP_K="4" \
    APP_PORT="8000"

EXPOSE 8000

CMD ["/bin/sh", "/app/scripts/start.sh"]
