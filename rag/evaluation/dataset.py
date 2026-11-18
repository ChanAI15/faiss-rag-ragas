"""Load and validate golden evaluation dataset."""

import json
from pathlib import Path
from typing import Any

from rag.config import EVAL_DATA_PATH


def load_evaluation_dataset(
    data_path: Path | str = EVAL_DATA_PATH,
    limit: int | None = None,
    answerable_only: bool = False,
) -> list[dict[str, Any]]:
    """Load evaluation questions and ground truth references from JSON."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")

    raw_data = json.loads(path.read_text(encoding="utf-8"))
    questions = raw_data.get("questions", [])

    if answerable_only:
        questions = [q for q in questions if q.get("answerable", True)]

    if limit is not None and limit > 0:
        questions = questions[:limit]

    return questions
