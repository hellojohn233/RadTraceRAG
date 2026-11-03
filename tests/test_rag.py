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
