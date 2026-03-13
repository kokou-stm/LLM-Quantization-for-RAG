import json
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    embeddings = np.array(embeddings, dtype=np.float32)
    return _normalize(embeddings)


def save_docstore(path: str, docs: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)


def load_docstore(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_faiss_index(path: str, index: faiss.IndexFlatIP) -> None:
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> faiss.IndexFlatIP:
    return faiss.read_index(path)


def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    docs: List[Dict],
    top_k: int = 4,
) -> List[Tuple[Dict, float]]:
    query_vec = embed_texts(model, [query])
    scores, indices = index.search(query_vec, top_k)
    results: List[Tuple[Dict, float]] = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        results.append((docs[int(idx)], float(score)))
    return results
