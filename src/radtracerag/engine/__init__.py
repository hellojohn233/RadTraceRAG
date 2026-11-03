"""通用 LightRAG 引擎：语料无关，与业务场景解耦。

本包只负责「配置 → 索引 → 检索 → 证据 → 回答」这条链路，
不包含任何领域知识或特定数据集的知识。
"""

from radtracerag.engine.settings import EngineSettings

__all__ = ["EngineSettings"]
