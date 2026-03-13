from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model
    model_path: str = "models/phi-3-mini-4k-instruct-q4.gguf"
    n_ctx: int = 4096
    n_threads: int = 8
    n_gpu_layers: int = 0  # set -1 on Mac Metal to offload all layers
    temperature: float = 0.2
    max_tokens: int = 512

    # RAG
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    faiss_index_path: str = "data/index.faiss"
    docstore_path: str = "data/docstore.json"
    rag_top_k: int = 4

    # API
    app_host: str = "0.0.0.0"
    app_port: int = 8000


settings = Settings()
