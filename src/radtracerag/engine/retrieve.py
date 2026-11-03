"""检索层：把 LightRAG 的结构化检索结果转成带编号的证据列表。

为什么用 ``aquery_data`` 而不是 ``aquery``
------------------------------------------
``aquery`` 内部调 ``aquery_llm``，只返回 ``llm_response["content"]`` ——
即使设了 ``only_need_context=True``，拿到的也只是渲染成字符串的 context
模板（引用块在末尾、正文是 JSON 块），还得自己解析字符串。

``aquery_data`` 直接返回结构化数据：

.. code-block:: python

    {"status": "success",
     "data": {"entities": [...], "relationships": [...],
              "chunks": [{"reference_id", "content", "file_path", "chunk_id"}],
              "references": [{"reference_id", "file_path"}]},
     "metadata": {"query_mode", "keywords", "processing_info"}}

⚠️ ``include_references`` 是 HTTP API 层（``api/routers/query_routes.py``）的
参数，SDK 路径根本不读它 —— 不要在 ``QueryParam`` 里设它。

⚠️ ``reference_id`` 是**文件级**的：LightRAG 按 ``file_path`` 去重后分配编号，
同一文件的多个 chunk 共用同一个 ID。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from lightrag import QueryParam

from radtracerag.engine.contracts import Evidence, RetrievalStep
from radtracerag.engine.errors import EngineError
from radtracerag.engine.settings import EngineSettings


@dataclass(frozen=True)
class RetrievalResult:
    """一次检索的结果。"""

    question: str
    mode: str
    evidences: tuple[Evidence, ...]
    steps: tuple[RetrievalStep, ...]
    metadata: dict[str, Any]


def evidence_from_payload(
    payload: dict[str, Any], path_map: dict[str, str]
) -> tuple[Evidence, ...]:
    """从 ``aquery_data`` 的返回结构构造证据列表。

    证据编号 ``E1`` / ``E2`` …… 按返回顺序生成，与 LightRAG 自己的
    ``reference_id``（文件级）区分开。

    ``path_map`` 的 key 是 basename、value 是完整路径 —— LightRAG 会把
    ``file_path`` 截成 basename，完整路径得我们自己还原。
    """
    data = payload.get("data") or {}
    chunks = data.get("chunks") or []

    evidences: list[Evidence] = []
    for index, chunk in enumerate(chunks, start=1):
        file_path = chunk.get("file_path", "") or ""
        evidences.append(
            Evidence(
                evidence_id=f"E{index}",
                chunk_id=chunk.get("chunk_id", "") or "",
                reference_id=str(chunk.get("reference_id", "") or ""),
                content=chunk.get("content", "") or "",
                source=path_map.get(file_path, file_path),
            )
        )
    return tuple(evidences)


async def retrieve(
    rag,
    question: str,
    mode: str,
    settings: EngineSettings,
    path_map: dict[str, str],
) -> RetrievalResult:
    """执行一次检索，返回证据与轨迹。

    ``mode`` 取 LightRAG 支持的检索模式，本项目对外只暴露
    ``naive`` / ``local`` / ``global`` / ``mix``。
    """
    param = QueryParam(
        mode=mode,
        enable_rerank=settings.enable_rerank,
        top_k=settings.top_k,
        chunk_top_k=settings.chunk_top_k,
    )

    started = time.perf_counter()
    payload = await rag.aquery_data(question, param)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # ⚠️ aquery_data **不抛异常**，它把查询失败编码进返回值里
    # （``{"status": "failure", "message": ...}``）。不检查的话，服务商故障
    # 会被显示成「没有检索到可用证据」，退出码还是 0 —— CI 会把坏掉的服务
    # 当成通过。
    #
    # 但「没查到结果」是正常结果（上游用 ``failure_reason="no_results"``
    # 标记它），不能也变成异常。
    if str(payload.get("status", "success")).lower() == "failure":
        metadata = payload.get("metadata") or {}
        failure_reason = str(metadata.get("failure_reason", ""))
        message = str(payload.get("message", "") or "")
        if failure_reason == "no_results" or "no results" in message.lower():
            pass  # 正常但没有命中，交给上层按「无证据」处理
        else:
            raise EngineError("检索失败", detail=message or failure_reason)

    evidences = evidence_from_payload(payload, path_map)
    metadata = dict(payload.get("metadata") or {})

    steps = (
        RetrievalStep(
            stage="检索",
            detail=f"{mode} 模式",
            elapsed_ms=elapsed_ms,
            count=len(evidences),
        ),
        RetrievalStep(
            stage="命中",
            detail=f"{metadata.get('query_mode', mode)} · 实体 "
            f"{len((payload.get('data') or {}).get('entities') or [])} · "
            f"关系 {len((payload.get('data') or {}).get('relationships') or [])}",
            count=len(evidences),
        ),
    )

    return RetrievalResult(
        question=question,
        mode=mode,
        evidences=evidences,
        steps=steps,
        metadata=metadata,
    )
