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
``lightrag.llm.openai`` 在 import 时读取模块级常量（如
``EMBEDDING_USE_BASE64``），所以 .env 必须在它之前加载。本模块把 openai
helper 的导入推迟到第一次调用时，确保 dotenv 已经生效。

（实测：``import lightrag`` 本身**不会**顺带 import ``lightrag.llm.openai``，
所以这道防线目前不是必需的 —— 保留它是因为上游哪天改成 eager import 也不会
把我们咬到。注释里原来的说法夸大了这个风险，这里按实测结果写。）
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import numpy as np
import tiktoken
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc

from radtracerag.engine.settings import PROJECT_ROOT, EngineSettings

#: 模型名 → tokenizer 的缓存。tiktoken 的 encoding_for_model 不便宜，
#: 而 embedding 一次会传进来几十上百条文本。
_ENCODINGS: dict[str, Any] = {}


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


def _get_encoding(model: str):
    """取模型的 tokenizer。

    返回 ``None`` 表示拿不到 —— 调用方据此退回字符估算，不让词表下载失败
    卡死整个摄入流程。

    拿不到有两种情况：

    1. **模型不认识**：不少 OpenAI 兼容服务跑的是非 OpenAI 模型
       （Qwen / DeepSeek / BGE 等），``tiktoken.encoding_for_model`` 直接
       抛 KeyError。
    2. **词表下载不了**：tiktoken 的 BPE 词表要联网取（默认缓存在临时目录，
       重启即失效）。网络受限的环境里这一步会失败。

    两种情况下都退回保守的字符估算：截断本来就是保守操作，
    宁可切短一点，也不要让服务端返回 400。
    """
    if model not in _ENCODINGS:
        encoding = None
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:  # pragma: no cover - 取决于网络与缓存状态
                encoding = None
        except Exception:  # pragma: no cover - 取决于网络与缓存状态
            encoding = None
        _ENCODINGS[model] = encoding
    return _ENCODINGS[model]


def _is_cjk(char: str) -> bool:
    """判断是不是中日韩字符 —— 它们的 token 密度比拉丁字母高得多。"""
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF      # 中日韩统一表意文字
        or 0x3040 <= code <= 0x30FF   # 日文假名
        or 0xAC00 <= code <= 0xD7AF   # 韩文谚文
        or 0x3400 <= code <= 0x4DBF   # 扩展 A
    )


def _estimate_tokens(text: str) -> int:
    """粗估 token 数。

    以 cl100k_base 的实测值为准：中文约 **2 token/字**，拉丁文约
    **4~8 字符/token**。这里取偏保守的一边（中文 2、拉丁 4），
    宁可估高一点提前截断，也不要估低了让服务端拒收。
    """
    cjk = sum(1 for char in text if _is_cjk(char))
    return cjk * 2 + (len(text) - cjk) // 4


def _truncate_by_estimate(text: str, max_token_size: int) -> str:
    """拿不到 tokenizer 时的保守截断。"""
    budget = max_token_size * 4  # 以 1/4 token 为单位记账
    out: list[str] = []
    spent = 0
    for char in text:
        cost = 8 if _is_cjk(char) else 1
        if spent + cost > budget:
            break
        spent += cost
        out.append(char)
    return "".join(out)


def truncate_for_embedding(
    texts: list[str], max_token_size: int | None, model: str
) -> list[str]:
    """把超过 token 上限的文本截断。

    ⚠️ LightRAG 的 ``EmbeddingFunc.__call__`` **只负责把 max_token_size 注入**
    给底层函数，**不会替我们截断** —— 截断是 embedding 函数自己的责任
    （LightRAG 自带的 ``openai_embed`` 就是在函数内部做的）。
    不截断的话，超长文本会被 embedding 服务端直接拒掉（400）。

    ``disallowed_special=()`` 是必须的：tiktoken 默认 ``disallowed_special=ALL``，
    文本里只要**出现**特殊标记字符串就抛 ValueError —— 一段引用了这类标记的
    内容会让整批 embedding 失败，跟它多长无关。
    """
    if not max_token_size or max_token_size <= 0:
        return texts

    encoding = _get_encoding(model)
    truncated: list[str] = []
    for text in texts:
        if not text:
            truncated.append(text)
            continue
        if encoding is None:
            truncated.append(
                _truncate_by_estimate(text, max_token_size)
                if _estimate_tokens(text) > max_token_size
                else text
            )
            continue
        try:
            tokens = encoding.encode(text, disallowed_special=())
        except Exception:  # pragma: no cover - 词表不可用时的兜底
            truncated.append(
                _truncate_by_estimate(text, max_token_size)
                if _estimate_tokens(text) > max_token_size
                else text
            )
            continue
        if len(tokens) > max_token_size:
            truncated.append(encoding.decode(tokens[:max_token_size]))
        else:
            truncated.append(text)
    return truncated


def build_embedding_func(settings: EngineSettings) -> EmbeddingFunc:
    """构造可直接传给 ``LightRAG(embedding_func=...)`` 的向量函数。

    内层协程的签名里**必须**带 ``max_token_size``：LightRAG 的
    ``EmbeddingFunc.__call__`` 会做签名检查，命中才把这个参数注入进来。
    注意 LightRAG **只注入、不截断** —— 截断得我们自己做，
    见 :func:`truncate_for_embedding`。
    """
    client = _make_client(settings)

    async def _embed(
        texts: list[str],
        max_token_size: int | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        # **kwargs 是刻意留的：LightRAG 官方 openai_embed 也是这个签名。
        # LightRAG 会传 context（asymmetric 场景）等参数，EmbeddingFunc 会
        # 弹掉不支持的，但将来多传一个参数也不该让整批 embedding 崩掉。
        payloads = truncate_for_embedding(
            list(texts), max_token_size, settings.embedding_model
        )
        response = await client.embeddings.create(
            model=settings.embedding_model,
            input=payloads,
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
