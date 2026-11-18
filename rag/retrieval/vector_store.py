"""FAISS index construction and persistence."""

import json

import faiss
import numpy as np

from rag.config import CHUNKS_PATH, INDEX_PATH
from rag.ingestion.embedder import embed_chunks
from rag.schemas import Chunk


STORAGE_DIR = INDEX_PATH.parent


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an inner-product FAISS index from normalized embeddings."""
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array")
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinity")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index: faiss.Index) -> None:
    """Persist a FAISS index under storage/index.faiss."""
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))


def load_index() -> faiss.Index:
    """Load the persisted FAISS index."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH}")
    return faiss.read_index(str(INDEX_PATH))


def save_chunks(chunks: list[Chunk]) -> None:
    vectors = embed_chunks([chunk.to_dict() for chunk in chunks])
    save_index(build_index(vectors))
    CHUNKS_PATH.write_text(json.dumps([chunk.to_dict() for chunk in chunks], ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks() -> list[dict[str, object]]:
    """Load chunk text and metadata aligned with the FAISS vector IDs."""
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("Chunk metadata not found. Run `python main.py ingest` first.")
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
