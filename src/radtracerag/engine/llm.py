"""组装 LLM 与 Embedding 调用函数。

为什么不直接用 LightRAG 自带的 ``openai_embed``
------------------------------------------------
LightRAG 1.5.7 的 ``openai_embed`` 有两个会让我们直接踩雷的地方：

1. 它被 ``@wrap_embedding_func_with_attrs(embedding_dim=1536)`` 装饰，
   **维度硬编码 1536**。用 1024 维的模型会抛
   ``ValueError: Embedding dimension mismatch detected``。
2. 它固定发送 ``encoding_format="base64"``（``llm/openai.py:1189``）。
   该开关是**模块级常量** ``EMBEDDING_USE_BASE64``，在 import 时读取；
   多数 OpenAI 兼容端点不支持 base64，会直接返回 400。

所以这里自己用 openai SDK 直连，显式指定 ``encoding_format="float"``，
维度由 :class:`EngineSettings` 控制。

关于导入顺序
------------
``lightrag.llm.openai`` 在 import 时读取模块级常量，因此 .env 必须先加载。
本模块把 openai helper 的导入推迟到第一次调用时，确保 dotenv 已经生效。
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import numpy as np
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc

from radtracerag.engine.settings import PROJECT_ROOT, EngineSettings


def _make_client(settings: EngineSettings):
    """创建 embedding 用的异步客户端。测试通过 monkeypatch 本函数替换。"""
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        base_url=settings.embedding_base_url,
        api_key=settings.embedding_api_key,
    )


def _load_openai_helpers():
    """加载 .env 后再导入 LightRAG 的 openai helper。

    必须延迟导入：``lightrag.llm.openai`` 在 import 时就会读取模块级常量
    （如 ``EMBEDDING_USE_BASE64``），dotenv 没加载的话读到的是空值。
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    from lightrag.llm.openai import openai_complete_if_cache

    return openai_complete_if_cache


def build_embedding_func(settings: EngineSettings) -> EmbeddingFunc:
    """构造可直接传给 ``LightRAG(embedding_func=...)`` 的向量函数。

    内层协程的签名里**必须**带 ``max_token_size``：LightRAG 的
    ``EmbeddingFunc.__call__`` 会做签名检查，命中才注入截断参数，
    超长文本才会被自动截断而不是被服务端拒绝。
    """
    client = _make_client(settings)

    async def _embed(texts: list[str], max_token_size: int | None = None) -> np.ndarray:
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=texts,
            encoding_format="float",
        )
        return np.asarray([item.embedding for item in response.data], dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=settings.embedding_dim,
        func=_embed,
        max_token_size=settings.embedding_max_token_size,
        model_name=settings.embedding_model,
    )


def build_llm_func(settings: EngineSettings) -> Callable[..., Awaitable[str]]:
    """构造可直接传给 ``LightRAG(llm_model_func=...)`` 的生成函数。"""

    complete = _load_openai_helpers()

    async def _llm(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        return await complete(
            model=settings.llm_model,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            **kwargs,
        )

    return _llm


async def embed_once(settings: EngineSettings, text: str) -> tuple[np.ndarray, int]:
    """调用一次 embedding，返回 ``(向量, 实际维度)``。

    ``doctor`` 用它核对服务商返回的真实维度与 ``EMBEDDING_DIM`` 是否一致 ——
    与其等到建索引时炸掉，不如在体检阶段就报出来。
    """
    client = _make_client(settings)
    response = await client.embeddings.create(
        model=settings.embedding_model,
        input=[text],
        encoding_format="float",
    )
    vector = np.asarray(response.data[0].embedding, dtype=np.float32)
    return vector, int(vector.shape[0])
