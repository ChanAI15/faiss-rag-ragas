"""Word-based text chunking with overlap."""

from rag.schemas import Chunk


def chunk_text(text: str, source: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Split text into word chunks; ``start`` stores the starting word offset."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")
    words = text.split()
    step = chunk_size - overlap
    return [
        Chunk(text=" ".join(words[start : start + chunk_size]), source=source, start=start)
        for start in range(0, len(words), step)
    ]
