"""Collect pipeline execution samples for evaluation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.config import EVAL_OUTPUT_DIR, TOP_K
from rag.generation.generator import generate_answer


def collect_evaluation_samples(
    questions: list[dict[str, Any]],
    top_k: int = TOP_K,
    save_cache: bool = True,
) -> tuple[list[dict[str, Any]], Path | None]:
    """Execute RAG pipeline for each question and format as RAGAS evaluation samples."""
    samples = []
    print(f"\nCollecting responses for {len(questions)} evaluation questions (top_k={top_k})...")

    for i, item in enumerate(questions, start=1):
        question_text = item["user_input"]
        print(f"[{i}/{len(questions)}] Querying: {question_text[:60]}...")
        gen_result = generate_answer(question_text, top_k=top_k)

        sample = {
            "id": item.get("id", f"q{i:02d}"),
            "user_input": question_text,
            "response": gen_result.get("answer", ""),
            "retrieved_contexts": gen_result.get("contexts", []),
            "reference": item.get("reference", ""),
            "reference_contexts": item.get("reference_contexts", []),
            "question_type": item.get("question_type", "general"),
            "sources": gen_result.get("sources", []),
        }
        samples.append(sample)

    cache_path = None
    if save_cache:
        EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_path = EVAL_OUTPUT_DIR / f"samples_{timestamp}.json"
        cache_path.write_text(
            json.dumps({"count": len(samples), "samples": samples}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved evaluation samples cache to {cache_path}")

    return samples, cache_path
