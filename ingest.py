import pickle
from pathlib import Path

import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, DATA_DIR, INDEX_DIR

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


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

    model = get_embedding_model()
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
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
