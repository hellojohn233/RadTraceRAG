"""演示数据的形状测试。

这组数据被界面和 API 共用，形状错了两边一起错——
之前就出过「引用编号被误当成指标对象」的事故，这里锁住。
"""

from __future__ import annotations

from radtracerag.contracts import Metric
from radtracerag.fixtures import MODES


def test_三组对照都在() -> None:
    assert set(MODES) == {"A · 无 RAG", "B · 向量 RAG", "C · 图增强 RAG"}


def test_引用编号是字符串元组() -> None:
    for spec in MODES.values():
        ids = spec["answer"].evidence_ids
        assert isinstance(ids, tuple), f"{ids} 不是元组"
        assert all(isinstance(i, str) for i in ids), f"{ids} 里有非字符串"


def test_指标是Metric对象() -> None:
    for spec in MODES.values():
        metrics = spec["metrics"]
        assert metrics, "每组都要有指标"
        assert all(isinstance(m, Metric) for m in metrics)


def test_图像观察的实体是字符串() -> None:
    for spec in MODES.values():
        entities = spec["observation"].entities
        assert isinstance(entities, tuple)
        assert all(isinstance(e, str) for e in entities)


def test_引用编号都能对上证据条目() -> None:
    for spec in MODES.values():
        available = {e.evidence_id for e in spec["evidence"]}
        for cited in spec["answer"].evidence_ids:
            assert cited in available, f"引用了不存在的证据 {cited}"
