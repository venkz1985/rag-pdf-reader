import pickle
from pathlib import Path

import fitz  # PyMuPDF
import faiss
import httpx
import numpy as np

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DATA_DIR, INDEX_DIR
from config import HF_TOKEN, HF_INFERENCE_URL


def get_embeddings(texts: list[str]) -> np.ndarray:
    response = httpx.post(
        f"{HF_INFERENCE_URL}{EMBEDDING_MODEL}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": texts},
        timeout=60.0,
    )
    response.raise_for_status()
    embeddings = np.array(response.json(), dtype="float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def extract_text(pdf_path: str | Path) -> str:
    doc = fitz.open(str(pdf_path))
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(text: str, source_file: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({
                "text": chunk.strip(),
                "source_file": source_file,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
        start = end - overlap
    return chunks


def build_index(pdf_paths: list[Path] | None = None) -> None:
    if pdf_paths is None:
        pdf_paths = list(DATA_DIR.glob("*.pdf"))

    if not pdf_paths:
        return

    all_chunks = []
    for pdf_path in pdf_paths:
        text = extract_text(pdf_path)
        chunks = chunk_text(text, source_file=pdf_path.name)
        all_chunks.extend(chunks)

    if not all_chunks:
        return

    texts = [c["text"] for c in all_chunks]
    embeddings = get_embeddings(texts)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(all_chunks, f)


def load_index() -> tuple[faiss.Index, list[dict]] | None:
    index_path = INDEX_DIR / "faiss.index"
    meta_path = INDEX_DIR / "metadata.pkl"

    if not index_path.exists() or not meta_path.exists():
        return None

    index = faiss.read_index(str(index_path))
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata
