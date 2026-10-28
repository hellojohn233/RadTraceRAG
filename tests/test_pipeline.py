"""pipeline 编排测试：不碰模型、不碰 Milvus，只验证链路和轨迹。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from radtracerag.contracts import (
    Answer,
    Evidence,
    ImageObservation,
    ImageQuestion,
    QueryMode,
    RetrievalStep,
)
from radtracerag.pipeline import (
    RetrievalOutcome,
    answer_question,
    build_query,
    build_terms,
)


class StubObserver:
    def __init__(self, observation: ImageObservation) -> None:
        self.observation = observation
        self.calls: list[tuple[Path, str]] = []

    async def observe(self, image_path: Path, question: str) -> ImageObservation:
        self.calls.append((image_path, question))
        return self.observation


class StubRetriever:
    def __init__(self, outcome: RetrievalOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[str, QueryMode]] = []

    async def retrieve(self, query: str, mode: QueryMode) -> RetrievalOutcome:
        self.calls.append((query, mode))
        return self.outcome


class StubAnswerer:
    def __init__(self, answer: Answer) -> None:
        self._answer = answer
        self.modes: list[QueryMode] = []

    async def answer(
        self,
        image_path: Path,
        question: str,
        observation: ImageObservation | None,
        evidence,
        mode: QueryMode,
    ) -> Answer:
        self.modes.append(mode)
        return self._answer


def _evidence() -> list[Evidence]:
    return [
        Evidence(evidence_id="1", content="肝是人体最大的腺体", source="SLAKE · 肝脏条目"),
        Evidence(evidence_id="2", content="胆汁由肝细胞分泌", source="SLAKE · 胆汁条目"),
    ]


QUESTION = ImageQuestion(image_path=Path("/tmp/ct.png"), question="图中主要器官的作用是什么？")
OBSERVATION = ImageObservation(text="腹部轴位断层", entities=("肝", "胆囊"))


def test_轨迹顺序固定() -> None:
    observer = StubObserver(OBSERVATION)
    retriever = StubRetriever(
        RetrievalOutcome(
            evidence=_evidence(),
            steps=[RetrievalStep("向量检索", "取 top-k", elapsed_ms=12, count=2)],
        )
    )
    answerer = StubAnswerer(Answer(text="答案 [1]", evidence_ids=("1",)))

    result = asyncio.run(
        answer_question(
            QUESTION,
            QueryMode.NAIVE,
            observer=observer,
            answerer=answerer,
            retriever=retriever,
        )
    )

    assert [step.stage for step in result.trace] == [
        "图像理解",
        "查询构造",
        "向量检索",
        "基于证据回答",
    ]
    assert result.answer.evidence_ids == ("1",)
    assert [(m.value, m.label) for m in result.metrics] == [("1", "引用证据"), ("2", "候选事实")]


def test_A组不检索() -> None:
    retriever = StubRetriever(RetrievalOutcome())
    observer = StubObserver(OBSERVATION)
    answerer = StubAnswerer(Answer(text="无引用", evidence_ids=()))

    result = asyncio.run(
        answer_question(
            QUESTION,
            QueryMode.NONE,
            observer=observer,
            answerer=answerer,
            retriever=retriever,
        )
    )

    assert retriever.calls == []
    assert result.evidence == []
    assert [step.stage for step in result.trace] == ["图像理解", "查询构造", "基于证据回答"]


def test_检索模式缺检索器会报错() -> None:
    with pytest.raises(ValueError):
        asyncio.run(
            answer_question(
                QUESTION,
                QueryMode.MIX,
                observer=StubObserver(OBSERVATION),
                answerer=StubAnswerer(Answer(text="x")),
                retriever=None,
            )
        )


def test_检索词优先用实体() -> None:
    assert build_terms(OBSERVATION) == ["肝", "胆囊"]
    assert build_query("作用是什么？", OBSERVATION) == "作用是什么？ 肝 胆囊"


def test_没有实体时退回观察文本() -> None:
    observation = ImageObservation(text="腹部轴位断层")
    assert build_terms(observation) == ["腹部轴位断层"]
    assert build_query("问题", observation) == "问题 腹部轴位断层"
