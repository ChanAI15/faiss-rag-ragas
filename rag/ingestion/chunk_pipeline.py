"""Create inspectable chunk JSON files from extracted per-page JSON."""

import json
from pathlib import Path

def chunk_extraction_file(path: Path, chunk_size: int, overlap: int) -> Path:
    """Chunk a complete document so overlap can continue across page boundaries."""
    extracted = json.loads(path.read_text(encoding="utf-8"))
    source = str(extracted["source"])
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words: list[str] = []
    page_ranges: list[tuple[int, int, int]] = []
    for page in extracted["pages"]:
        page_words = str(page["text"]).split()
        start = len(words)
        words.extend(page_words)
        if page_words:
            page_ranges.append((int(page["page_number"]), start, len(words) - 1))

    def page_for_word(word_index: int) -> int:
        for page_number, start, end in page_ranges:
            if start <= word_index <= end:
                return page_number
        raise ValueError(f"Word index {word_index} is outside the extracted page ranges")

    step = chunk_size - overlap
    chunks = [
        {
            "chunk_number": chunk_number,
            "start_word": start,
            "end_word": end - 1,
            "page_start": page_for_word(start),
            "page_end": page_for_word(end - 1),
            "text": " ".join(words[start:end]),
        }
        for chunk_number, start in enumerate(range(0, len(words), step), start=1)
        if (end := min(start + chunk_size, len(words))) > start
    ]

    output_path = path.with_name(f"{path.stem}.chunks.json")
    output_path.write_text(
        json.dumps(
            {
                "source": source,
                "chunk_size_words": chunk_size,
                "overlap_words": overlap,
                "step_size_words": chunk_size - overlap,
                "chunk_count": len(chunks),
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path
