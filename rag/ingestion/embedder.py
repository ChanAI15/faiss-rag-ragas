"""Gemini Embedding 2 helpers."""

import numpy as np
from google.genai import types

from rag.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL, client, require_api_key


def _embed(text: str) -> np.ndarray:
    require_api_key()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
    )
    return np.asarray(result.embeddings[0].values, dtype=np.float32)


def embed_chunks(chunks: list[dict[str, object]]) -> np.ndarray:
    """Embed and normalize document chunks for FAISS inner-product search."""
    if not chunks:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    require_api_key()
    contents = [
        types.Content(parts=[types.Part.from_text(text=f"title: none | text: {chunk['text']}")])
        for chunk in chunks
    ]
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=contents,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIMENSION),
    )
    embeddings = np.asarray([item.values for item in response.embeddings], dtype=np.float32)
    return embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)


def embed_document(text: str) -> np.ndarray:
    return _embed(f"title: none | text: {text}")


def embed_query(question: str) -> np.ndarray:
    return _embed(f"task: question answering | query: {question}")
