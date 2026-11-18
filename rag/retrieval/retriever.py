"""Retrieval orchestration and formatted context assembly."""

from rag.config import TOP_K
from rag.retrieval.search import search_documents


def retrieve_context(
    question: str,
    top_k: int = TOP_K,
) -> dict:
    results = search_documents(
        question=question,
        top_k=top_k,
    )

    if not results:
        return {
            "question": question,
            "chunks": [],
            "context": "",
            "sources": [],
        }

    context_parts = []
    sources = []

    for result in results:
        source = {
            "document": result["source"],
            "pages": result["page_numbers"],
            "chunk_id": result["chunk_id"],
        }

        context_parts.append(
            f"[Source: {source['document']}, "
            f"Pages: {', '.join(map(str, source['pages']))}]\n"
            f"{result['text']}"
        )

        sources.append(source)

    return {
        "question": question,
        "chunks": results,
        "context": "\n\n".join(context_parts),
        "sources": sources,
    }


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve raw ranked chunk dictionaries (backward-compatibility helper)."""
    return retrieve_context(question, top_k)["chunks"]
