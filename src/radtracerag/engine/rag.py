"""构造 LightRAG 实例，并管理存储的生命周期。

存储全部用 LightRAG 的默认实现（JsonKV / NanoVectorDB / NetworkX /
JsonDocStatus）—— **不引入任何外部服务**，索引目录就是一个本地文件夹。

``rag_session`` 用 try/finally 保证异常时也会落盘，否则一次失败的 ingest
可能把索引写坏。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from lightrag import LightRAG
from lightrag.utils import setup_logger

from radtracerag.engine.llm import build_embedding_func, build_llm_func
from radtracerag.engine.settings import EngineSettings

#: LightRAG 的日志名
LOGGER_NAME = "lightrag"


def build_rag(settings: EngineSettings) -> LightRAG:
    """按配置构造一个 LightRAG 实例（尚未初始化存储）。

    日志级别用 ``setup_logger`` 压到 WARNING —— LightRAG 1.5.7 已废弃
    ``log_level`` 构造参数（传入会告警并被 delattr）。
    """
    setup_logger(LOGGER_NAME, level="WARNING", enable_file_logging=False)
    return LightRAG(
        working_dir=str(settings.working_dir),
        llm_model_func=build_llm_func(settings),
        llm_model_name=settings.llm_model,
        embedding_func=build_embedding_func(settings),
        top_k=settings.top_k,
        chunk_top_k=settings.chunk_top_k,
        llm_model_max_async=settings.llm_max_async,
        default_llm_timeout=settings.llm_timeout,
    )


@asynccontextmanager
async def rag_session(settings: EngineSettings) -> AsyncIterator[LightRAG]:
    """打开一个可用��� LightRAG 会话，退出时必定落盘。"""
    rag = build_rag(settings)
    await rag.initialize_storages()
    try:
        yield rag
    finally:
        await rag.finalize_storages()
