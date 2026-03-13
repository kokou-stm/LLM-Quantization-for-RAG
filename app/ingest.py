import argparse
import re
from typing import List, Dict

import requests
from sentence_transformers import SentenceTransformer

from .rag import embed_texts, build_faiss_index, save_faiss_index, save_docstore
from .settings import settings


def fetch_wikipedia_page(title: str, lang: str = "en") -> str:
    url = f"https://{lang}.wikipedia.org/w/api.php"
    headers = {"User-Agent": "quantized-rag/0.1 (local test; contact: dev@example.com)"}
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "format": "json",
    }
    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return ""
    page = next(iter(pages.values()))
    return page.get("extract", "")


def chunk_text(text: str, chunk_size: int = 350, overlap: int = 40) -> List[str]:
    words = re.findall(r"\S+", text)
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_docs(titles: List[str], lang: str = "en") -> List[Dict]:
    docs: List[Dict] = []
    for title in titles:
        text = fetch_wikipedia_page(title, lang=lang)
        for i, chunk in enumerate(chunk_text(text)):
            docs.append(
                {
                    "id": f"{title}:{i}",
                    "title": title,
                    "source": f"https://{lang}.wikipedia.org/wiki/{title}",
                    "text": chunk,
                }
            )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from Wikipedia pages")
    parser.add_argument(
        "--pages",
        required=True,
        help="Comma-separated list of Wikipedia page titles, e.g. 'Azure,Large_language_model'",
    )
    parser.add_argument("--lang", default="en", help="Wikipedia language (default: en)")
    parser.add_argument("--out-index", default=settings.faiss_index_path)
    parser.add_argument("--out-docs", default=settings.docstore_path)
    args = parser.parse_args()

    titles = [p.strip().replace(" ", "_") for p in args.pages.split(",") if p.strip()]
    if not titles:
        raise SystemExit("No pages provided")

    docs = build_docs(titles, lang=args.lang)
    embedder = SentenceTransformer(settings.embed_model)
    embeddings = embed_texts(embedder, [d["text"] for d in docs])

    index = build_faiss_index(embeddings)
    save_faiss_index(args.out_index, index)
    save_docstore(args.out_docs, docs)

    print(f"Saved {len(docs)} chunks")
    print(f"Index: {args.out_index}")
    print(f"Docstore: {args.out_docs}")


if __name__ == "__main__":
    main()
