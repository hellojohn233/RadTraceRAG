import logging
import warnings

import pytest

from radtracerag.engine.rag import build_rag, rag_session


def test_build_rag_passes_expected_arguments(settings):
    with warnings.catch_warnings():
        # 传了已废弃的 log_level / log_file_path 参数会在这里变成异常
        warnings.simplefilter("error", UserWarning)
        rag = build_rag(settings)
    assert rag.working_dir == str(settings.working_dir)
    assert rag.llm_model_name == settings.llm_model
    assert rag.embedding_func.embedding_dim == settings.embedding_dim
    assert rag.top_k == settings.top_k
    assert rag.chunk_top_k == settings.chunk_top_k
    assert rag.log_level is None  # 未传废弃参数
    assert logging.getLogger("lightrag").level == logging.WARNING


class _FakeRag:
    def __init__(self, events):
        self.events = events

    async def initialize_storages(self):
        self.events.append("init")

    async def finalize_storages(self):
        self.events.append("final")


@pytest.mark.asyncio
async def test_rag_session_initializes_and_finalizes(monkeypatch, settings):
    events = []
    monkeypatch.setattr("radtracerag.engine.rag.build_rag", lambda s: _FakeRag(events))
    async with rag_session(settings) as rag:
        assert isinstance(rag, _FakeRag)
        assert events == ["init"]
    assert events == ["init", "final"]


@pytest.mark.asyncio
async def test_rag_session_finalizes_on_error(monkeypatch, settings):
    events = []
    monkeypatch.setattr("radtracerag.engine.rag.build_rag", lambda s: _FakeRag(events))
    with pytest.raises(RuntimeError):
        async with rag_session(settings):
            raise RuntimeError("boom")
    assert events == ["init", "final"]


def test_build_rag_silences_third_party_loggers(settings):
    """第三方 logger 也要压到 WARNING。

    ``nano-vectordb`` 是**独立的顶层 logger**（不是 ``lightrag`` 的子 logger），
    压 ``lightrag`` 对它无效 —— 每次跑 CLI 都会冒三行 ``INFO:nano-vectordb``。
    LightRAG 自己还会把 ``openai`` 的 logger 设成 INFO。
    """
    build_rag(settings)
    # nano_vectordb 在 import 时调用 logging.basicConfig(level=INFO)，
    # 把 **root** logger 抬到了 INFO —— 所有 NOTSET 的第三方 logger 都跟着
    # 变成 INFO。不压 root 就压不干净。
    assert logging.getLogger().level == logging.WARNING
    # openai 3.42.1 用的是 httpx2 包，它的 logger 名叫 "httpx2"
    for name in ("nano-vectordb", "openai", "httpx2", "httpcore2"):
        assert logging.getLogger(name).level == logging.WARNING, name
