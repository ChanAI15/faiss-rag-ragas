"""Shared data structures."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    start: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
