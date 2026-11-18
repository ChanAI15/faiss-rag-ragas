"""Similarity search against the persisted FAISS index."""

import json
from pathlib import Path

import numpy as np

from rag.config import CHUNKS_PATH as CONFIG_CHUNKS_PATH
from rag.ingestion.embedder import embed_query
from rag.retrieval.vector_store import load_index

CHUNKS_PATH = CONFIG_CHUNKS_PATH if CONFIG_CHUNKS_PATH.exists() else Path("storage/chunks.json")


def search_documents(
    question: str,
    top_k: int = 4,
) -> list[dict]:
    if not question.strip():
        raise ValueError("Question cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    index = load_index()

    if index.ntotal == 0:
        return []

    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    query_vector = embed_query(question)

    query_vector = np.ascontiguousarray(
        query_vector.reshape(1, -1),
        dtype=np.float32,
    )

    if query_vector.shape[1] != index.d:
        raise ValueError("Query and index dimensions differ")

    scores, ids = index.search(
        query_vector,
        min(top_k, index.ntotal),
    )

    results = []

    for vector_id, score in zip(ids[0], scores[0]):
        if vector_id < 0:
            continue

        chunk = chunks[int(vector_id)]
        p_start = chunk.get("page_start", chunk.get("page_number", 1))
        p_end = chunk.get("page_end", p_start)
        page_numbers = chunk.get(
            "page_numbers",
            [p_start] if p_start == p_end else list(range(p_start, p_end + 1)),
        )

        chunk_id = chunk.get("chunk_id", chunk.get("chunk_number", int(vector_id) + 1))

        results.append({
            **chunk,
            "source": chunk.get("source", "ares_dawn.pdf"),
            "page_numbers": page_numbers,
            "chunk_id": chunk_id,
            "start": chunk.get("start", chunk.get("start_word", 0)),
            "vector_id": int(vector_id),
            "score": float(score),
        })

    return results


search = search_documents
