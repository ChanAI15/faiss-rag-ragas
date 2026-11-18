"""Result ranking hook for optional future reranking."""


def rank(results: list[dict[str, object]]) -> list[dict[str, object]]:
    """Order FAISS candidates by descending similarity score."""
    return sorted(results, key=lambda item: float(item["score"]), reverse=True)
