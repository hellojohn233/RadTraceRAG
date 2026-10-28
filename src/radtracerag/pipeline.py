"""核心问答链路。传输层无关。

CLI、Streamlit、FastAPI 都只调 `answer_question()`——换传输层不动这里，
换模型只换注入的 adapter。界面上那条叙事流就是这里 `trace` 的顺序：

    图像理解 → 查询构造 → 检索 → 基于证据回答
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

from radtracerag.contracts import (
    Answer,
    Evidence,
    ImageObservation,
    ImageQuestion,
    Metric,
    QueryMode,
    RetrievalStep,
)

Clock = Callable[[], float]


class VisionObserver(Protocol):
    """看片：只产出与当前问题有关的观察，不是整张图的泛化描述。"""

    async def observe(self, image_path: Path, question: str) -> ImageObservation: ...


class Retriever(Protocol):
    """检索：交回证据，以及这一步本身的轨迹。"""

    async def retrieve(self, query: str, mode: QueryMode) -> "RetrievalOutcome": ...


class Answerer(Protocol):
    """基于证据作答：必须回传引用到的证据 ID，不能凭空编号。

    `mode` 也要传——三组对照用的是不同提示词版本，跑批时要记下这一条，
    才说得清某个结果是哪一组、哪版提示词跑出来的。
    """

    async def answer(
        self,
        image_path: Path,
        question: str,
        observation: ImageObservation | None,
        evidence: Sequence[Evidence],
        mode: QueryMode,
    ) -> Answer: ...


@dataclass(frozen=True)
class RetrievalOutcome:
    evidence: list[Evidence] = field(default_factory=list)
    steps: list[RetrievalStep] = field(default_factory=list)


@dataclass(frozen=True)
class AnswerResult:
    """一次问答的完整结果：答案，以及它凭什么这么说。"""

    answer: Answer
    observation: ImageObservation | None = None
    evidence: list[Evidence] = field(default_factory=list)
    trace: list[RetrievalStep] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    mode: QueryMode = QueryMode.MIX


def build_terms(observation: ImageObservation | None) -> list[str]:
    """从图像观察里取检索词：实体优先，没有实体就用观察文本。"""
    if observation is None:
        return []
    if observation.entities:
        return list(observation.entities)
    return [observation.text.strip()] if observation.text.strip() else []


def build_query(question: str, observation: ImageObservation | None) -> str:
    """问题 + 图像观察里的实体，拼成检索语句。"""
    terms = build_terms(observation)
    if not terms:
        return question.strip()
    return f"{question.strip()} {' '.join(terms)}"


def _elapsed_ms(start: float, clock: Clock) -> int:
    return int((clock() - start) * 1000)


async def answer_question(
    question: ImageQuestion,
    mode: QueryMode,
    *,
    observer: VisionObserver,
    answerer: Answerer,
    retriever: Retriever | None = None,
    clock: Clock = time.perf_counter,
) -> AnswerResult:
    """问一次。返回答案、证据、检索轨迹和指标。

    `mode` 是三组对照里的哪一组：A（无 RAG）不检索，B 向量，C 图增强。
    """
    trace: list[RetrievalStep] = []

    # 1. 看片
    start = clock()
    observation = await observer.observe(question.image_path, question.question)
    trace.append(
        RetrievalStep(
            stage="图像理解",
            detail="VLM 读取影像，产出与问题相关的观察",
            elapsed_ms=_elapsed_ms(start, clock),
        )
    )

    # 2. 查询构造
    start = clock()
    terms = build_terms(observation)
    query = build_query(question.question, observation)
    trace.append(
        RetrievalStep(
            stage="查询构造",
            detail=(
                f"把图像观察中的实体拼成检索词：{' / '.join(terms)}"
                if terms
                else "图像观察没有给出可用实体，只用原问题检索"
            ),
            elapsed_ms=_elapsed_ms(start, clock),
        )
    )

    # 3. 检索（A 组不检索）
    evidence: list[Evidence] = []
    if mode is not QueryMode.NONE:
        if retriever is None:
            raise ValueError(f"{mode.label} 需要检索器，但没传 retriever")
        outcome = await retriever.retrieve(query, mode)
        evidence = list(outcome.evidence)
        trace.extend(outcome.steps)

    # 4. 基于证据作答
    start = clock()
    answer = await answerer.answer(
        question.image_path, question.question, observation, evidence, mode
    )
    trace.append(
        RetrievalStep(
            stage="基于证据回答",
            detail="原图 + 证据生成答案，并回传引用 ID",
            elapsed_ms=_elapsed_ms(start, clock),
        )
    )

    metrics = [
        Metric(str(len(answer.evidence_ids)), "引用证据"),
        Metric(str(len(evidence)), "候选事实"),
    ]
    return AnswerResult(
        answer=answer,
        observation=observation,
        evidence=evidence,
        trace=trace,
        metrics=metrics,
        mode=mode,
    )
