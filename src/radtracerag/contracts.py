"""Data shapes shared across the future image, retrieval, and answer stages."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageQuestion:
    image_path: Path
    question: str


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    content: str
    source: str


@dataclass(frozen=True)
class Answer:
    text: str
    evidence_ids: tuple[str, ...] = ()
