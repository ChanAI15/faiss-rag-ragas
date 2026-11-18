# RAG Retrieval & Context Assembly Issues & Improvements

This document tracks identified areas of improvement and proposed architectural optimizations for `rag/retrieval/retriever.py` when assembling context and metadata for LLM answer generation.

---

## 1. Identified Issues & Opportunities

### Issue 1: Fragmented Document Reading Order in Context
- **Current Behavior**: Chunks in `context` are ordered strictly by descending similarity score (e.g., Chunk 8 [score: 0.82], Chunk 2 [score: 0.79], Chunk 7 [score: 0.75]).
- **Impact**: When multiple chunks from the same document are retrieved, they appear out of sequence. LLMs perform substantially better when narrative and logical flow follow natural chronological document order (`chunk_id` or `start_word`).
- **Proposed Solution**: Maintain score-sorted order in `chunks` (for inspection and ranking), but sort the rendered `context` passages by original document position (`source`, `chunk_id`).

---

### Issue 2: Duplicate Overlap Sentences in Context
- **Current Behavior**: The chunking pipeline applies an 80-word overlap across consecutive chunks.
- **Impact**: If adjacent chunks (e.g., Chunk 4 and Chunk 5) are both retrieved, the 80 overlapping words appear twice within the prompt context. This wastes prompt tokens and presents repeated sentences to the LLM.
- **Proposed Solution**: Detect adjacent chunks (`chunk_id == prev_chunk_id + 1` from the same source document) and stitch them together by trimming the repeated overlapping prefix.

---

### Issue 3: Absence of Relevance Score Thresholding
- **Current Behavior**: `retrieve_context` always returns `top_k` chunks regardless of their similarity score.
- **Impact**: When a user asks an off-topic or out-of-domain question, low-similarity distractor chunks (e.g., score < 0.25) are still fed into the prompt, increasing the risk of LLM hallucinations.
- **Proposed Solution**: Introduce an optional `score_threshold: float | None = 0.3` parameter to discard low-confidence noise before building the prompt context.

---

### Issue 4: Sparse Source Metadata for Downstream Consumers
- **Current Behavior**: The `sources` array only contains:
  ```json
  {
    "document": "ares_dawn.pdf",
    "pages": [1, 2],
    "chunk_id": 1
  }
  ```
- **Impact**: Downstream consumer applications, UI components, and evaluation harnesses lack:
  - Similarity score for confidence meters.
  - Human-friendly page labels (`Page 1` vs `Pages 1-2`).
- **Proposed Solution**: Enrich the `sources` metadata dictionary with `score` and formatted `page_label`.

---

### Issue 5: Context Demarcation & LLM Citation Structure
- **Current Behavior**: Context blocks are separated by `[Source: ..., Pages: ...]`.
- **Impact**: LLMs (such as Gemini) attend and cite more reliably when passages are marked with structured Markdown headings or explicit block labels (e.g., `### [Passage 1] - ares_dawn.pdf (Pages 1-2) | Relevance: 0.8234`).
- **Proposed Solution**: Use structured Markdown passage headings with clear metadata tags.

---

## 2. Proposed Implementation for `retriever.py`

```python
"""Retrieval orchestration and formatted context assembly."""

from rag.config import TOP_K
from rag.retrieval.search import search_documents


def format_page_range(pages: list[int]) -> str:
    """Format [1, 2] as 'Pages 1-2' and [1] as 'Page 1'."""
    if not pages:
        return "Unknown Page"
    if len(pages) == 1:
        return f"Page {pages[0]}"
    return f"Pages {min(pages)}-{max(pages)}"


def retrieve_context(
    question: str,
    top_k: int = TOP_K,
    score_threshold: float | None = None,
    order_by_document: bool = True,
) -> dict:
    results = search_documents(question=question, top_k=top_k)

    # Optional: Filter out low-relevance noise
    if score_threshold is not None:
        results = [r for r in results if r.get("score", 0.0) >= score_threshold]

    if not results:
        return {
            "question": question,
            "chunks": [],
            "context": "",
            "sources": [],
        }

    # Order context passages by document position for coherent reading
    ordered_results = (
        sorted(results, key=lambda r: (str(r.get("source")), r.get("chunk_id", 0)))
        if order_by_document
        else results
    )

    context_parts = []
    sources = []

    for index, result in enumerate(ordered_results, start=1):
        pages = result.get("page_numbers", [])
        page_str = format_page_range(pages)
        source_doc = result.get("source", "unknown")
        chunk_id = result.get("chunk_id")
        score = round(float(result.get("score", 0.0)), 4)

        sources.append({
            "document": source_doc,
            "page_numbers": pages,
            "page_label": page_str,
            "chunk_id": chunk_id,
            "score": score,
        })

        context_parts.append(
            f"### [Passage {index}] - {source_doc} ({page_str}) | Relevance: {score}\n"
            f"{result['text'].strip()}"
        )

    return {
        "question": question,
        "chunks": results,  # preserves rank-order by score
        "context": "\n\n".join(context_parts),
        "sources": sources,
    }


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve raw ranked chunk dictionaries (backward-compatibility helper)."""
    return retrieve_context(question, top_k)["chunks"]
```

---

## 3. Action Items Checklist

- [ ] Add `format_page_range()` helper for clean citation labels.
- [ ] Add `score_threshold` parameter for noise rejection.
- [ ] Implement `order_by_document` sorting for context passage coherence.
- [ ] Add `score` and `page_label` to each item in `sources`.
- [ ] Implement adjacent chunk overlap deduplication/stitching.
