"""Data shapes shared across the future image, retrieval, and answer stages."""

from dataclasses import dataclass
from enum import Enum
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


@dataclass(frozen=True)
class Metric:
    """一行细排的指标，不做成卡片块。"""

    value: str
    label: str


class QueryMode(str, Enum):
    """三组对照。值直接对应 LightRAG 的检索模式，便于前后端共用同一套枚举。"""

    NONE = "none"
    NAIVE = "naive"
    MIX = "mix"

    @property
    def label(self) -> str:
        return {
            QueryMode.NONE: "无 RAG",
            QueryMode.NAIVE: "向量 RAG",
            QueryMode.MIX: "图增强 RAG",
        }[self]

    @property
    def group(self) -> str:
        """对照组编号，界面上用来标 A / B / C。"""
        return {
            QueryMode.NONE: "A",
            QueryMode.NAIVE: "B",
            QueryMode.MIX: "C",
        }[self]

    @property
    def desc(self) -> str:
        return {
            QueryMode.NONE: "只看影像和问题，让模型直接回答",
            QueryMode.NAIVE: "LightRAG naive 模式，纯文本向量检索",
            QueryMode.MIX: "LightRAG mix 模式，图关系 + 向量联合检索",
        }[self]

    @classmethod
    def parse(cls, value: str | None) -> "QueryMode":
        """容忍大小写和 A/B/C 这类组号写法。"""
        if not value:
            return cls.MIX
        key = value.strip().lower()
        aliases = {
            "a": cls.NONE,
            "b": cls.NAIVE,
            "c": cls.MIX,
            "none": cls.NONE,
            "no_rag": cls.NONE,
            "naive": cls.NAIVE,
            "vector": cls.NAIVE,
            "mix": cls.MIX,
            "graph": cls.MIX,
        }
        if key not in aliases:
            raise ValueError(f"未知检索模式：{value}")
        return aliases[key]
