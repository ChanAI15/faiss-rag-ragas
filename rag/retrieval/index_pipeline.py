"""Build a persisted FAISS index from inspectable embedding JSON."""

import json
from pathlib import Path

import numpy as np

from rag.config import CHUNKS_PATH
from rag.retrieval.vector_store import build_index, save_index


def index_embedding_file(path: Path) -> int:
    """Create storage/index.faiss from one saved embedding JSON file."""
    embedding_document = json.loads(path.read_text(encoding="utf-8"))
    entries = list(embedding_document["embeddings"])
    embeddings = np.asarray([entry["embedding"] for entry in entries], dtype=np.float32)
    save_index(build_index(embeddings))
    chunks = [{key: value for key, value in entry.items() if key != "embedding"} for entry in entries]
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(chunks)
