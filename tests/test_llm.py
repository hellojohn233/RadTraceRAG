import inspect

import numpy as np
import pytest

from radtracerag.engine.llm import build_embedding_func, build_llm_func, embed_once


class _FakeEmbeddings:
    def __init__(self, captured, dim=768):
        self._captured = captured
        self._dim = dim

    async def create(self, **kwargs):
        self._captured.update(kwargs)
        row = type("D", (), {"embedding": [0.1] * self._dim})()
        return type("R", (), {"data": [row]})()


class _FakeClient:
    def __init__(self, captured, dim=768):
        self.embeddings = _FakeEmbeddings(captured, dim)


@pytest.mark.asyncio
async def test_embedding_func_declares_max_token_size(settings):
    func = build_embedding_func(settings)
    assert func.embedding_dim == 1024
    assert func.max_token_size == 8192
    # 内层协程签名必须含 max_token_size，LightRAG 靠签名检查注入截断参数
    assert "max_token_size" in inspect.signature(func.func).parameters


@pytest.mark.asyncio
async def test_embed_once_returns_actual_dimension(monkeypatch, settings):
    captured = {}
    monkeypatch.setattr(
        "radtracerag.engine.llm._make_client", lambda s: _FakeClient(captured)
    )
    vec, dim = await embed_once(settings, "hi")
    assert dim == 768
    assert vec.shape == (768,)
    assert captured["encoding_format"] == "float"  # 关键：不能是 base64
    assert captured["model"] == settings.embedding_model
    assert captured["input"] == ["hi"]


@pytest.mark.asyncio
async def test_embed_once_uses_configured_base_url(monkeypatch, settings):
    seen = {}

    def _fake_make(s):
        seen["base_url"] = s.embedding_base_url
        seen["api_key"] = s.embedding_api_key
        return _FakeClient({})

    monkeypatch.setattr("radtracerag.engine.llm._make_client", _fake_make)
    await embed_once(settings, "hi")
    assert seen["base_url"] == settings.embedding_base_url
    assert seen["api_key"] == settings.embedding_api_key


def test_build_llm_func_returns_coroutine_function(settings):
    func = build_llm_func(settings)
    assert callable(func)
    assert inspect.iscoroutinefunction(func)


@pytest.mark.asyncio
async def test_embedding_func_truncates_overlong_text(monkeypatch, settings_factory):
    """超长文本必须在发出去之前被截断。

    LightRAG 的 ``EmbeddingFunc.__call__`` **只负责把 max_token_size 注入**
    给底层函数，**不会替我们截断** —— 截断是 embedding 函数自己的责任
    （LightRAG 自带的 ``openai_embed`` 就是在函数内部做的）。不截断的话，
    超长 chunk 会被服务端直接拒掉（400）。
    """
    captured = {}
    monkeypatch.setattr(
        "radtracerag.engine.llm._make_client",
        lambda s: _FakeClient(captured, dim=1024),
    )
    settings = settings_factory(EMBEDDING_MAX_TOKEN_SIZE=10)
    func = build_embedding_func(settings)

    long_text = "肝" * 500
    await func([long_text])
    assert captured["input"][0] != long_text
    assert len(captured["input"][0]) < len(long_text)


@pytest.mark.asyncio
async def test_embedding_func_keeps_short_text_intact(monkeypatch, settings_factory):
    """没超限的文本不能被截断 —— 截断只对超限的那部分生效。"""
    captured = {}
    monkeypatch.setattr(
        "radtracerag.engine.llm._make_client",
        lambda s: _FakeClient(captured, dim=1024),
    )
    settings = settings_factory(EMBEDDING_MAX_TOKEN_SIZE=8192)
    func = build_embedding_func(settings)

    await func(["短文本"])
    assert captured["input"] == ["短文本"]


@pytest.mark.asyncio
async def test_embedding_func_truncates_without_tokenizer(monkeypatch, settings_factory):
    """拿不到 tokenizer（离线 / 词表下载失败）时也要截断，不能直接放行超长文本。"""
    monkeypatch.setattr(
        "radtracerag.engine.llm._get_encoding", lambda model: None
    )
    captured = {}
    monkeypatch.setattr(
        "radtracerag.engine.llm._make_client",
        lambda s: _FakeClient(captured, dim=1024),
    )
    settings = settings_factory(EMBEDDING_MAX_TOKEN_SIZE=10)
    func = build_embedding_func(settings)

    long_text = "肝" * 500
    await func([long_text])
    assert captured["input"][0] != long_text
    assert len(captured["input"][0]) <= 10 * 2  # 中文 2 token/字
