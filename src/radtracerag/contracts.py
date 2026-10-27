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


@dataclass(frozen=True)
class ImageObservation:
    """与当前问题相关的图像观察，不是整张图的泛化描述。"""

    text: str
    entities: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalStep:
    """检索轨迹的一步，用于把「答案怎么来的」摊开给人看。"""

    stage: str
    detail: str
    elapsed_ms: int = 0
    count: int = 0
