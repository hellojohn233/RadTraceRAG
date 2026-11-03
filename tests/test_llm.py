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
