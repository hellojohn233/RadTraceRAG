"""引擎通用的数据结构。

这些类型刻意保持"纯数据"：不含业务语义，不含任何领域字段，
因此可以被任意语料、任意场景复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    """一份待索引的原始文档。"""

    path: Path
    doc_id: str
    media_type: str


@dataclass(frozen=True)
class Evidence:
    """一条检索证据。

    ⚠️ 没有 ``score`` 字段：LightRAG 的任何公开返回路径都不提供相似度分数，
    与其编造一个，不如不提供。
    """

    evidence_id: str  # 我们自己的编号，形如 E1 / E2
    chunk_id: str  # LightRAG 的 chunk 唯一 ID
    reference_id: str  # LightRAG 的文件级编号，同一文件的 chunk 共用
    content: str
    source: str  # 完整路径


@dataclass(frozen=True)
class RetrievalStep:
    """检索轨迹的一步，用于把"答案怎么来的"摊开给人看。"""

    stage: str
    detail: str
    elapsed_ms: int = 0
    count: int = 0


@dataclass(frozen=True)
class Answer:
    """一次基于证据的回答。

    ``unsupported_markers`` 非空表示模型引用了不存在的证据编号 ——
    这些编号已从正文中移除，这里留下痕迹以便观测。
    """

    text: str
    evidence_ids: tuple[str, ...] = ()
    unsupported_markers: tuple[str, ...] = ()
