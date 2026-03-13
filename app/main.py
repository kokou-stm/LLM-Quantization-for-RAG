from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from .settings import settings
from .rag import load_docstore, load_faiss_index, retrieve


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]


app = FastAPI(title="Quantized LLM + RAG")

llm: Optional[Llama] = None
embedder: Optional[SentenceTransformer] = None
rag_index = None
rag_docs = None


@app.on_event("startup")
def load_resources() -> None:
    global llm, embedder, rag_index, rag_docs

    model_path = Path(settings.model_path)
    if not model_path.exists():
        raise RuntimeError(
            f"Model not found at {model_path}. Set MODEL_PATH env var or download a GGUF model."
        )

    llm = Llama(
        model_path=str(model_path),
        n_ctx=settings.n_ctx,
        n_threads=settings.n_threads,
        n_gpu_layers=settings.n_gpu_layers,
    )

    embedder = SentenceTransformer(settings.embed_model)

    index_path = Path(settings.faiss_index_path)
    docs_path = Path(settings.docstore_path)
    if index_path.exists() and docs_path.exists():
        rag_index = load_faiss_index(str(index_path))
        rag_docs = load_docstore(str(docs_path))
    else:
        rag_index = None
        rag_docs = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if llm is None or embedder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    context_blocks = []
    sources: List[Dict[str, str]] = []
    if rag_index is not None and rag_docs is not None:
        results = retrieve(req.question, embedder, rag_index, rag_docs, settings.rag_top_k)
        for doc, score in results:
            context_blocks.append(f"[Source] {doc['text']}")
            sources.append({"title": doc.get("title", ""), "source": doc.get("source", "")})

    system_prompt = (
        "You are a helpful assistant. Use the provided context to answer. "
        "If the answer is not in the context, say you do not know."
    )
    context = "\n\n".join(context_blocks) if context_blocks else ""

    prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{req.question}\n"
    if context:
        prompt += f"<|context|>\n{context}\n"
    prompt += "<|assistant|>\n"

    output = llm(
        prompt,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        stop=["<|user|>", "<|assistant|>", "<|system|>", "</s>"],
    )

    answer = output["choices"][0]["text"].strip()
    return ChatResponse(answer=answer, sources=sources)
