"""构造 LightRAG 实例，并管理存储的生命周期。

存储全部用 LightRAG 的默认实现（JsonKV / NanoVectorDB / NetworkX /
JsonDocStatus）—— **不引入任何外部服务**，索引目录就是一个本地文件夹。

``rag_session`` 用 try/finally 保证异常时也会落盘，否则一次失败的 ingest
可能把索引写坏。
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import logging

from lightrag import LightRAG
from lightrag.utils import setup_logger

from radtracerag.engine.llm import build_embedding_func, build_llm_func
from radtracerag.engine.settings import EngineSettings

#: LightRAG 的日志名
LOGGER_NAME = "lightrag"

#: 需要一起压到 WARNING 的第三方 logger。
#:
#: ``nano-vectordb`` 是**独立的顶层 logger**（不是 ``lightrag`` 的子 logger），
#: 只压 ``lightrag`` 对它无效。更要紧的是它在 import 时调用了
#: ``logging.basicConfig(level=logging.INFO)``（``nano_vectordb/dbs.py:13``）——
#: 这会把 **root** logger 抬到 INFO，于是所有 NOTSET 的第三方 logger 都跟着
#: 变成 INFO。所以必须连 root 一起压。
#:
#: ⚠️ 这个 import 发生在 ``LightRAG.__init__`` **内部**，所以压制必须在
#: 构造**之后**再执行一次 —— 详见 :func:`build_rag`。
#:
#: ``openai`` 3.42.1 用的是 **httpx2** 包，logger 名叫 ``"httpx2"``；
#: LightRAG 自己还会把 ``openai`` 的 logger 设成 INFO（``llm/openai.py:411``）。
THIRD_PARTY_LOGGERS = (
    "nano-vectordb",
    "openai",
    "httpx2",
    "httpcore2",
    "httpx",
    "httpcore",
)


def _silence_third_party_loggers() -> None:
    """把第三方 logger（**含 root**）压到 WARNING。

    root 必须一起压：``nano_vectordb`` 在 import 时调用
    ``logging.basicConfig(level=logging.INFO)``，把所有 NOTSET 的第三方
    logger 都抬成 INFO，只压具名的几个不够。
    """
    setup_logger(LOGGER_NAME, level="WARNING", enable_file_logging=False)
    logging.getLogger().setLevel(logging.WARNING)
    for name in THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def build_rag(settings: EngineSettings) -> LightRAG:
    """按配置构造一个 LightRAG 实例（尚未初始化存储）。

    日志级别用 ``setup_logger`` 压到 WARNING —— LightRAG 1.5.7 已废弃
    ``log_level`` 构造参数（传入会告警并被 delattr）。
    """
    # 构造前先压一次：让构造过程本身保持安静。
    _silence_third_party_loggers()

    rag = LightRAG(
        working_dir=str(settings.working_dir),
        llm_model_func=build_llm_func(settings),
        llm_model_name=settings.llm_model,
        embedding_func=build_embedding_func(settings),
        top_k=settings.top_k,
        chunk_top_k=settings.chunk_top_k,
        llm_model_max_async=settings.llm_max_async,
        default_llm_timeout=settings.llm_timeout,
    )

    # ⚠️ 构造**之后**必须再压一次。
    #
    # ``LightRAG.__init__`` 内部会 import ``nano_vectordb``
    # （``lightrag/kg/nano_vector_db_impl.py:23``），而后者在 import 时
    # 调用 ``logging.basicConfig(level=logging.INFO)``
    # （``nano_vectordb/dbs.py:13``）—— 这会把 **root** logger 抬回 INFO。
    #
    # 之前只在构造前压，等于被构造过程推翻了：真实进程里 root 最终是
    # INFO(20)，CLI 输出会混进第三方库的 INFO 日志。而当时的守卫测试
    # 在 pytest 下永远是绿的 —— pytest 的 logging 插件预装了 root handler，
    # ``basicConfig`` 变成空操作，压根不会抬 level。测试护不住它。
    _silence_third_party_loggers()
    return rag


@asynccontextmanager
async def rag_session(settings: EngineSettings) -> AsyncIterator[LightRAG]:
    """打开一个可用的 LightRAG 会话，退出时必定落盘。"""
    rag = build_rag(settings)
    await rag.initialize_storages()
    try:
        yield rag
    finally:
        await rag.finalize_storages()
