"""Pydantic 模型：HTTP 契约的唯一定义处。

前端用 `openapi-typescript` 从这里生成 TS 类型，前后端不会各写各的。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from radtracerag.contracts import QueryMode


class MetricOut(BaseModel):
    value: str
    label: str


class AnswerOut(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class ObservationOut(BaseModel):
    text: str
    entities: list[str] = Field(default_factory=list)


class EvidenceOut(BaseModel):
    evidence_id: str
    content: str
    source: str


class TraceStepOut(BaseModel):
    stage: str
    detail: str
    elapsed_ms: int = 0
    count: int = 0


class QueryResultOut(BaseModel):
    """一次问答的完整结果：答案，以及它凭什么这么说。"""

    mode: str
    mode_group: str
    mode_label: str
    answer: AnswerOut
    observation: ObservationOut | None = None
    evidence: list[EvidenceOut] = Field(default_factory=list)
    trace: list[TraceStepOut] = Field(default_factory=list)
    metrics: list[MetricOut] = Field(default_factory=list)


class ModeOut(BaseModel):
    id: str
    group: str
    label: str
    desc: str


class HealthOut(BaseModel):
    status: str
    version: str
    engine: str
    engine_note: str
    workspace: str
    vector_storage: str
    graph_storage: str
    milvus_uri: str
    milvus_db_name: str
    modes: list[ModeOut]


def mode_payload() -> list[ModeOut]:
    return [
        ModeOut(
            id=mode.value,
            group=mode.group,
            label=mode.label,
            desc=mode.desc,
        )
        for mode in QueryMode
    ]
