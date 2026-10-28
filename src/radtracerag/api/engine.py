"""装配 pipeline 需要的三个 adapter。

两条路：

- **真实引擎**：模型服务和 Milvus 索引都配齐才启用
- **演示引擎**：形状完全一样，内容是合成样例，前端可以照着它开发

演示引擎不是「假的真答案」——它跑的是同一套 pipeline 编排，只是三个 adapter
换成固定样例。换句话说，接口契约是可信的，医学内容不是。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from radtracerag.config import Settings
from radtracerag.contracts import (
    Answer,
    Evidence,
    ImageObservation,
    QueryMode,
)
from radtracerag.fixtures import MODES
from radtracerag.pipeline import RetrievalOutcome

FIXTURE_KEY = {
    QueryMode.NONE: "A · 无 RAG",
    QueryMode.NAIVE: "B · 向量 RAG",
    QueryMode.MIX: "C · 图增强 RAG",
}

# 演示引擎里，检索阶段负责的那几步
RETRIEVAL_STAGES = {"向量检索", "图扩展", "命中"}


@dataclass
class Engine:
    """一组 adapter。`kind` 决定前端看到的是真实链路还是演示数据。"""

    observer: object
    answerer: object
    retriever: object | None
    kind: str
    note: str = ""


class DemoObserver:
    """演示用的看片：返回固定的合成观察。"""

    async def observe(self, image_path: Path, question: str) -> ImageObservation:
        return MODES[FIXTURE_KEY[QueryMode.MIX]]["observation"]


class DemoRetriever:
    """演示用的检索：按对照组给出候选证据和命中情况。"""

    async def retrieve(self, query: str, mode: QueryMode) -> RetrievalOutcome:
        if mode is QueryMode.NONE:
            return RetrievalOutcome()
        spec = MODES[FIXTURE_KEY[mode]]
        evidence = list(spec["evidence"])
        steps = [s for s in spec["trace"] if s.stage in RETRIEVAL_STAGES]
        return RetrievalOutcome(evidence=evidence, steps=steps)


class DemoAnswerer:
    """演示用的作答：按对照组给出固定答案。"""

    async def answer(
        self,
        image_path: Path,
        question: str,
        observation: ImageObservation | None,
        evidence: list[Evidence],
        mode: QueryMode,
    ) -> Answer:
        return MODES[FIXTURE_KEY[mode]]["answer"]


def build_demo_engine() -> Engine:
    return Engine(
        observer=DemoObserver(),
        answerer=DemoAnswerer(),
        retriever=DemoRetriever(),
        kind="demo",
        note="演示引擎：未配置模型服务，返回合成样例，不是真实医学结论",
    )


def model_configured() -> bool:
    """模型服务是否配齐。视觉模型、文本模型、嵌入模型各算一半。"""
    return all(
        os.environ.get(key)
        for key in (
            "RADTRACERAG_LLM_BASE_URL",
            "RADTRACERAG_LLM_API_KEY",
            "RADTRACERAG_VLM_BASE_URL",
            "RADTRACERAG_VLM_API_KEY",
            "RADTRACERAG_EMBEDDING_BASE_URL",
            "RADTRACERAG_EMBEDDING_API_KEY",
        )
    )


def build_engine(settings: Settings) -> Engine:
    """按配置选引擎。没配齐模型就退回演示引擎，并说明原因。"""
    if not model_configured():
        return build_demo_engine()

    from radtracerag.integrations.lightrag import LightRAGRetriever, build_rag
    from radtracerag.integrations.models import (
        ChatClient,
        EvidenceAnswerer,
        VLMObserver,
        build_embedding_func,
    )

    rag = build_rag(
        settings,
        llm_model_func=_build_llm_func(),
        embedding_func=build_embedding_func(),
    )
    return Engine(
        observer=VLMObserver(
            ChatClient(
                os.environ["RADTRACERAG_VLM_BASE_URL"],
                os.environ["RADTRACERAG_VLM_API_KEY"],
                os.environ.get("RADTRACERAG_VLM_MODEL", "qwen2.5-vl"),
            )
        ),
        answerer=EvidenceAnswerer(
            ChatClient(
                os.environ["RADTRACERAG_LLM_BASE_URL"],
                os.environ["RADTRACERAG_LLM_API_KEY"],
                os.environ.get("RADTRACERAG_LLM_MODEL", "qwen2.5"),
            )
        ),
        retriever=LightRAGRetriever(rag),
        kind="lightrag",
        note="真实链路：LightRAG + Milvus",
    )


def _build_llm_func():
    """LightRAG 需要的文本补全函数，走 OpenAI 兼容接口。"""
    from functools import partial

    from lightrag.llm.openai import openai_complete_if_cache

    return partial(
        openai_complete_if_cache,
        model=os.environ.get("RADTRACERAG_LLM_MODEL", "qwen2.5"),
        base_url=os.environ["RADTRACERAG_LLM_BASE_URL"],
        api_key=os.environ["RADTRACERAG_LLM_API_KEY"],
    )


__all__ = [
    "Engine",
    "DemoAnswerer",
    "DemoObserver",
    "DemoRetriever",
    "build_demo_engine",
    "build_engine",
    "model_configured",
]
