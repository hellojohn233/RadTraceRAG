from pathlib import Path

import pytest

from radtracerag.cli import main


class _FakeRag:
    def __init__(self):
        self.inserted = []

    async def ainsert(self, text, ids=None, file_paths=None, **kwargs):
        self.inserted.append({"text": text, "ids": ids, "file_paths": file_paths})
        return "mock-track-id"


def _fake_settings_cls(tmp_path, missing=()):
    """返回一个 EngineSettings 的替身类：from_env 后 missing_keys 报出缺项。"""

    class _FakeSettings:
        working_dir = Path(tmp_path)
        top_k = 40
        chunk_top_k = 20
        enable_rerank = False
        llm_model = "fake-model"
        embedding_model = "fake-emb"
        embedding_dim = 1024
        llm_base_url = "https://llm.example/v1"
        embedding_base_url = "https://emb.example/v1"
        llm_api_key = "sk-fake"
        embedding_api_key = "sk-fake"
        llm_max_async = 4
        llm_timeout = 240
        embedding_max_token_size = 8192

        @classmethod
        def from_env(cls, env_file=None):
            return cls()

        def missing_keys(self):
            return list(missing)

    return _FakeSettings


@pytest.fixture
def fake_rag_session(monkeypatch):
    """把 CLI 里的 rag_session 换成假的，避免真的建索引目录。"""
    import contextlib

    @contextlib.asynccontextmanager
    async def _fake(settings):
        yield _FakeRag()

    monkeypatch.setattr("radtracerag.cli.rag_session", _fake)


def test_no_subcommand_exits_with_usage_error():
    assert main([]) == 2


def test_invalid_mode_lists_valid_modes(capsys):
    rc = main(["query", "肝做什么", "--mode", "bogus"])
    captured = capsys.readouterr()
    assert rc == 2
    text = captured.out + captured.err
    for mode in ("naive", "local", "global", "mix"):
        assert mode in text


def test_query_missing_env_reports_missing_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "radtracerag.cli.EngineSettings",
        _fake_settings_cls(tmp_path, missing=["LLM_API_KEY", "EMBEDDING_MODEL"]),
    )
    rc = main(["query", "肝做什么"])
    captured = capsys.readouterr()
    assert rc == 1
    text = captured.out + captured.err
    assert "LLM_API_KEY" in text
    assert "EMBEDDING_MODEL" in text


def test_doctor_missing_env_returns_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "radtracerag.cli.EngineSettings", _fake_settings_cls(tmp_path, missing=["LLM_MODEL"])
    )
    rc = main(["doctor"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "LLM_MODEL" in captured.out + captured.err


def test_ingest_empty_dir_returns_zero(tmp_path, monkeypatch, capsys, fake_rag_session):
    corpus = tmp_path / "empty"
    corpus.mkdir()
    monkeypatch.setattr("radtracerag.cli.EngineSettings", _fake_settings_cls(tmp_path))
    assert main(["ingest", str(corpus)]) == 0


def test_ingest_missing_dir_returns_one(tmp_path, monkeypatch, capsys, fake_rag_session):
    monkeypatch.setattr("radtracerag.cli.EngineSettings", _fake_settings_cls(tmp_path))
    rc = main(["ingest", str(tmp_path / "nope")])
    captured = capsys.readouterr()
    assert rc == 1
    assert "不存在" in captured.out + captured.err


def test_eval_reports_not_built(capsys):
    rc = main(["eval"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "评测集" in captured.out + captured.err


def test_doctor_reports_actual_embedding_dimension_on_mismatch(
    tmp_path, monkeypatch, capsys
):
    """服务商的真实维度与配置不符时，必须报出**实际**维度并退出非 0。

    这是 review focus 里点名要钉住的一条 —— 之前行为是对的但没有测试，
    随时可能悄悄回归。
    """
    import numpy as np

    monkeypatch.setattr("radtracerag.cli.EngineSettings", _fake_settings_cls(tmp_path))

    async def _fake_embed_once(settings, text):
        return np.zeros(768, dtype=np.float32), 768

    monkeypatch.setattr("radtracerag.cli.embed_once", _fake_embed_once)

    rc = main(["doctor"])
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert rc == 1
    assert "768" in text, "要报出服务商的真实维度"
    assert "1024" in text, "也要报出配置的维度"
    assert "维度不一致" in text
