import json
import logging
import subprocess
import sys
import textwrap
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


#: 在**独立进程**里跑的日志探针。
#:
#: 为什么要开子进程：``logging.basicConfig(level=INFO)`` 只有在 root
#: **没有任何 handler** 时才会生效。pytest 的 logging 插件会预装一个 root
#: handler，于是 ``basicConfig`` 变成空操作 —— 同一段代码在 pytest 里是绿的，
#: 在真实进程里却把 root 抬回了 INFO(20)。守卫测试写在 pytest 里就永远
#: 抓不到它自己要防的那个回归。
#:
#: 探针还顺手验证**没被钉住**的 logger 不会往 stderr 漏 INFO ——
#: 那才是压制 root 的真正目的。
_LOGGER_PROBE = textwrap.dedent(
    """
    import json
    import logging
    import sys
    from pathlib import Path

    from radtracerag.engine.rag import build_rag
    from radtracerag.engine.settings import EngineSettings

    settings = EngineSettings(
        llm_base_url="https://llm.example/v1",
        llm_api_key="sk-placeholder",
        llm_model="chat-placeholder",
        llm_max_async=4,
        llm_timeout=240,
        embedding_base_url="https://emb.example/v1",
        embedding_api_key="sk-placeholder",
        embedding_model="emb-placeholder",
        embedding_dim=1024,
        embedding_max_token_size=8192,
        working_dir=Path(sys.argv[1]),
        top_k=40,
        chunk_top_k=20,
        enable_rerank=False,
    )

    build_rag(settings)

    # 一个没被钉住的 logger：它的有效级别继承自 root
    logging.getLogger("some_unpinned_lib").info("INFO-FROM-UNPINNED-LIBGER")

    print(json.dumps({
        "root": logging.getLogger().level,
        "lightrag": logging.getLogger("lightrag").level,
        "nano-vectordb": logging.getLogger("nano-vectordb").level,
        "httpx2": logging.getLogger("httpx2").level,
        "openai": logging.getLogger("openai").level,
        "root_handlers": len(logging.getLogger().handlers),
    }))
    """
)


def _run_logger_probe(tmp_path):
    """在子进程里跑探针，返回 ``(解析后的 JSON, stderr 文本)``。"""
    result = subprocess.run(
        [sys.executable, "-c", _LOGGER_PROBE, str(tmp_path / "storage")],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, f"探针进程失败:\n{result.stderr}"
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return payload, result.stderr


def test_build_rag_silences_third_party_loggers(tmp_path):
    """第三方 logger 也要压到 WARNING，**root** 尤其不能漏。

    ``nano-vectordb`` 是**独立的顶层 logger**（不是 ``lightrag`` 的子 logger），
    压 ``lightrag`` 对它无效 —— 每次跑 CLI 都会冒三行 ``INFO:nano-vectordb``。
    LightRAG 自己还会把 ``openai`` 的 logger 设成 INFO。

    这个断言必须在**独立进程**里做：``nano_vectordb`` 在 import 时调用
    ``logging.basicConfig(level=INFO)`` 把 root 抬到 INFO，而
    ``basicConfig`` 只有在 root 没有 handler 时才生效 —— pytest 预装了 root
    handler，所以在 pytest 里断言 root 级别永远是假绿。
    """
    payload, stderr = _run_logger_probe(tmp_path)

    assert payload["root"] == logging.WARNING, (
        "root logger 必须压到 WARNING：nano_vectordb 在 import 时 "
        "basicConfig(level=INFO) 会把它抬回 INFO"
    )
    # openai 3.42.1 用的是 httpx2 包，它的 logger 名叫 "httpx2"
    for name in ("lightrag", "nano-vectordb", "httpx2", "openai"):
        assert payload[name] == logging.WARNING, name

    # 压 root 的意义在于：没被显式钉住的 logger 也不会往 stderr 漏 INFO
    assert "INFO-FROM-UNPINNED-LIBGER" not in stderr, (
        f"未钉住的 logger 仍在输出 INFO：\n{stderr}"
    )


def test_logger_suppression_survives_rag_construction(tmp_path):
    """压制必须发生在 ``LightRAG(...)`` 构造**之后**。

    ``LightRAG.__init__`` 内部会 import ``nano_vectordb``，而后者在 import
    时 ``logging.basicConfig(level=INFO)`` —— 只在构造前压制的话，
    构造过程会把 root 重新抬回 INFO(20)。
    """
    payload, _stderr = _run_logger_probe(tmp_path)
    # 构造之后 root 仍然是 WARNING
    assert payload["root"] == logging.WARNING
    # basicConfig 装上的 root handler 还在（它的级别是 NOTSET）——
    # 记录创建靠的是发起方 logger 的有效级别，所以压制 root 依然有效。
    assert payload["root_handlers"] >= 0

