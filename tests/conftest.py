"""所有测试共享的构造器。无网络、无 API key。"""
from pathlib import Path

import pytest

from radtracerag.engine.settings import ENGINE_ENV_KEYS, EngineSettings

FULL_ENV = """\
LLM_BASE_URL=https://llm.example/v1
LLM_API_KEY=sk-llm-placeholder
LLM_MODEL=chat-placeholder
EMBEDDING_BASE_URL=https://emb.example/v1
EMBEDDING_API_KEY=sk-emb-placeholder
EMBEDDING_MODEL=emb-placeholder
EMBEDDING_DIM=1024
"""


def write_full_env(tmp_path: Path, **overrides: str) -> Path:
    """写一份完整 .env，返回其路径。overrides 覆盖个别变量。"""
    body = FULL_ENV + "".join(f"{k}={v}\n" for k, v in overrides.items())
    env_file = tmp_path / ".env"
    env_file.write_text(body, encoding="utf-8")
    return env_file


@pytest.fixture
def settings(tmp_path: Path) -> EngineSettings:
    return EngineSettings.from_env(write_full_env(tmp_path))


@pytest.fixture
def settings_factory(tmp_path: Path):
    """返回一个函数：传 overrides 得到对应 EngineSettings。"""
    def _make(**overrides) -> EngineSettings:
        return EngineSettings.from_env(write_full_env(tmp_path, **overrides))
    return _make


@pytest.fixture(autouse=True)
def _isolate_engine_env(monkeypatch):
    """清掉进程里可能存在的真实配置，保证测试只看到 .env 文件里的值。"""
    for key in ENGINE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
