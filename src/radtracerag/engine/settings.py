"""引擎配置：读取 .env，校验完整性，解析工作目录。

设计约定
--------
- **LLM 与 Embedding 是两组完全独立的配置**，各自有 base_url / api_key / model。
  原因：很多 OpenAI 兼容服务只提供 chat，不提供 embedding；反向的也存在。
- **不采用** LightRAG 自带的 ``LLM_BINDING_*`` / ``EMBEDDING_BINDING_*`` 约定。
  我们自己读、自己组装，语义更清楚，也不会和 LightRAG 内部常量抢导入时机。
- 读取优先级：**进程环境变量 > .env 文件**。这样 CI 或临时调试可以用
  ``LLM_MODEL=xxx uv run ...`` 覆盖文件里的值。
- ``working_dir`` 一律相对**项目根**解析，不跟随进程 CWD —— 否则换个目录跑
  就会指向另一个索引目录，表现为「检索不到」。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

# src/radtracerag/engine/settings.py -> parents[3] 即仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[3]

#: 引擎读取的全部环境变量名。进程环境变量里出现同名键时优先于 .env 文件。
ENGINE_ENV_KEYS = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_MAX_ASYNC",
    "LLM_TIMEOUT",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_MAX_TOKEN_SIZE",
    "RAG_WORKING_DIR",
    "RAG_TOP_K",
    "RAG_CHUNK_TOP_K",
    "RAG_ENABLE_RERANK",
)

#: 必填项，顺序即 ``missing_keys()`` 的返回顺序。
REQUIRED_KEYS = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIM",
)

DEFAULTS = {
    "LLM_MAX_ASYNC": 4,
    "LLM_TIMEOUT": 240,
    "EMBEDDING_MAX_TOKEN_SIZE": 8192,
    "RAG_WORKING_DIR": "rag_storage",
    "RAG_TOP_K": 40,
    "RAG_CHUNK_TOP_K": 20,
    "RAG_ENABLE_RERANK": False,
}

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _int(raw: dict[str, str], key: str) -> int:
    """取整数。缺失或空串 → 返回 0；有值但非法 → 抛 ValueError 并带上变量名。"""
    value = raw.get(key)
    if value is None or value.strip() == "":
        return 0
    try:
        return int(value.strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"环境变量 {key} 不是合法整数: {value!r}") from exc


def _bool(raw: dict[str, str], key: str) -> bool:
    value = raw.get(key)
    if value is None or value.strip() == "":
        return bool(DEFAULTS.get(key, False))
    return value.strip().lower() in _TRUE_VALUES


@dataclass(frozen=True)
class EngineSettings:
    """引擎的全部运行配置。"""

    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_max_async: int
    llm_timeout: int
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_dim: int
    embedding_max_token_size: int
    working_dir: Path
    top_k: int
    chunk_top_k: int
    enable_rerank: bool

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "EngineSettings":
        """从 .env 文件 + 进程环境变量构造配置。

        缺失的必填项不会在这里抛错 —— 它们会被记为 0 或空串，
        由 :meth:`missing_keys` 报告，这样 ``doctor`` 能一次性列出所有缺项。
        """
        path = Path(env_file) if env_file is not None else PROJECT_ROOT / ".env"
        raw: dict[str, str] = {}
        if path.exists():
            raw.update({k: v for k, v in dotenv_values(path).items() if v is not None})
        # 进程环境变量优先
        raw.update({k: v for k, v in os.environ.items() if k in ENGINE_ENV_KEYS})

        working_dir_value = raw.get("RAG_WORKING_DIR") or DEFAULTS["RAG_WORKING_DIR"]
        working_dir = (PROJECT_ROOT / working_dir_value).resolve()

        return cls(
            llm_base_url=raw.get("LLM_BASE_URL", "").strip(),
            llm_api_key=raw.get("LLM_API_KEY", "").strip(),
            llm_model=raw.get("LLM_MODEL", "").strip(),
            llm_max_async=_int(raw, "LLM_MAX_ASYNC") or int(DEFAULTS["LLM_MAX_ASYNC"]),
            llm_timeout=_int(raw, "LLM_TIMEOUT") or int(DEFAULTS["LLM_TIMEOUT"]),
            embedding_base_url=raw.get("EMBEDDING_BASE_URL", "").strip(),
            embedding_api_key=raw.get("EMBEDDING_API_KEY", "").strip(),
            embedding_model=raw.get("EMBEDDING_MODEL", "").strip(),
            embedding_dim=_int(raw, "EMBEDDING_DIM"),
            embedding_max_token_size=(
                _int(raw, "EMBEDDING_MAX_TOKEN_SIZE")
                or int(DEFAULTS["EMBEDDING_MAX_TOKEN_SIZE"])
            ),
            working_dir=working_dir,
            top_k=_int(raw, "RAG_TOP_K") or int(DEFAULTS["RAG_TOP_K"]),
            chunk_top_k=_int(raw, "RAG_CHUNK_TOP_K") or int(DEFAULTS["RAG_CHUNK_TOP_K"]),
            enable_rerank=_bool(raw, "RAG_ENABLE_RERANK"),
        )

    def missing_keys(self) -> list[str]:
        """返回缺失或非法的必填变量名，按 LLM_* → EMBEDDING_* 排序。"""
        missing: list[str] = []
        for key in REQUIRED_KEYS:
            if key == "EMBEDDING_DIM":
                if self.embedding_dim <= 0:
                    missing.append(key)
                continue
            if not getattr(self, key.lower()):
                missing.append(key)
        return missing
