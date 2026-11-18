"""Create inspectable embedding JSON files from chunk JSON."""

import json
from pathlib import Path

from rag.config import EMBEDDING_DIMENSION, EMBEDDING_MODEL
from rag.ingestion.embedder import embed_chunks


def embed_chunk_file(path: Path) -> Path:
    """Embed a chunk JSON file and write chunk metadata plus vectors to JSON."""
    chunk_document = json.loads(path.read_text(encoding="utf-8"))
    chunks = list(chunk_document["chunks"])
    embeddings = embed_chunks(chunks)
    entries = [
        {**chunk, "embedding": embedding.tolist()}
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    output_name = path.name.removesuffix(".chunks.json") + ".embeddings.json"
    output_path = path.with_name(output_name)
    output_path.write_text(
        json.dumps(
            {
                "source": chunk_document["source"],
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "embedding_count": len(entries),
                "embeddings": entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path
