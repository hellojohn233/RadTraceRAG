"""LightRAG integration point; model configuration is supplied by callers.

向量库走 LightRAG 自带的 Milvus 存储实现，图也用 LightRAG 的——
本项目不另建图。MILVUS_* 由 LightRAG 在初始化时从环境变量读，
官方 lightrag-server 也能用同一套变量指到同一个 Milvus。
"""

from __future__ import annotations

import time
from typing import Literal

from lightrag import LightRAG, QueryParam

from radtracerag.config import Settings
from radtracerag.contracts import Evidence, QueryMode, RetrievalStep
from radtracerag.pipeline import RetrievalOutcome

QueryModeParam = Literal["naive", "mix"]


async def query_context(
    rag: LightRAG, question: str, mode: QueryModeParam = "mix"
) -> str:
    """Query the same LightRAG index with either planned baseline mode."""
    return await rag.aquery(question, param=QueryParam(mode=mode, only_need_context=True))


def build_rag(
    settings: Settings,
    *,
    llm_model_func,
    embedding_func,
    rerank_model_func=None,
) -> LightRAG:
    """构造 LightRAG 实例。

    存储后端全部走配置：向量默认 Milvus，KV / 图 / 文档状态用本地实现。
    模型函数由调用方注入——本项目不绑定任何一家模型服务。
    """
    return LightRAG(
        working_dir=str(settings.storage_dir),
        workspace=settings.workspace,
        vector_storage=settings.vector_storage,
        kv_storage=settings.kv_storage,
        graph_storage=settings.graph_storage,
        doc_status_storage=settings.doc_status_storage,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        rerank_model_func=rerank_model_func,
    )


class LightRAGRetriever:
    """把 LightRAG 的检索接成 pipeline 需要的 Retriever。

    已知缺口：现在把 LightRAG 组装好的上下文当作一条证据返回，
    还没有把上下文拆回逐条 evidence_id。拆分要等索引里真的有内容、
    且能对上原始三元组之后才能做——在那之前宁可少报结构，也不要编编号。
    """

    def __init__(self, rag: LightRAG) -> None:
        self.rag = rag

    async def retrieve(self, query: str, mode: QueryMode) -> RetrievalOutcome:
        if mode is QueryMode.NONE:
            return RetrievalOutcome()

        param_mode: QueryModeParam = "naive" if mode is QueryMode.NAIVE else "mix"
        stage = "向量检索" if mode is QueryMode.NAIVE else "图扩展检索"

        start = time.perf_counter()
        context = await query_context(self.rag, query, mode=param_mode)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        evidence = (
            [
                Evidence(
                    evidence_id="1",
                    content=context,
                    source=f"LightRAG · {param_mode}",
                )
            ]
            if context.strip()
            else []
        )
        steps = [
            RetrievalStep(
                stage=stage,
                detail=f"LightRAG {param_mode} 模式",
                elapsed_ms=elapsed_ms,
                count=len(evidence),
            )
        ]
        return RetrievalOutcome(evidence=evidence, steps=steps)
