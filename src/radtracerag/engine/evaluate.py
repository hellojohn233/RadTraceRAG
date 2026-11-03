"""检索指标计算 —— 骨架。

本轮只定死接口并用内置样本做自检，**不产出任何对外指标数字**。
评测集、标注规则、统计显著性检验留到后续轮次（见设计文档 §7）。

指标定义
--------
输入是「查询 → 检索结果的有序列表」和「查询 → 相关项集合」两份映射：

- ``retrievals``：查询 ID → 检索到的 item ID 列表，**按相关度降序**
- ``gold``：查询 ID → 相关 item ID 的集合
- ``k``：只看前 k 个结果的窗口

返回的三个指标：

``recall@k``
    窗口内召回的相关结果比例：``|top-k ∩ gold| / |gold|``。
    检索不到任何相关结果时为 0。

``ndcg@k``
    归一化折损累计增益。DCG 按 ``1/log2(rank+1)`` 给每个命中的相关结果加权，
    IDCG 是「相关结果全都排在最前面」时的理想值，所以取值在 0 到 1 之间。

``mrr``
    首个相关结果的倒数排名的平均值（窗口内）。窗口内没有相关结果时为 0。
    这是 reciprocal rank 的标准算法；这里固定在 k 的窗口内计算，
    避免「第一个相关结果排在第 37 位」这种在有限窗口里没意义的命中
    贡献出一个极小的数值。

只统计 ``gold`` 里出现过的查询 —— 没有标注的查询没法评。
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def _dcg(hits: Sequence[bool], k: int) -> float:
    """折损累计增益。``hits[i]`` 表示第 i+1 位是否命中相关结果。"""
    total = 0.0
    for rank, hit in enumerate(hits[:k], start=1):
        if hit:
            total += 1.0 / math.log2(rank + 1)
    return total


def evaluate(
    retrievals: Mapping[str, Sequence[str]],
    gold: Mapping[str, Sequence[str]],
    k: int = 10,
) -> dict[str, float | int]:
    """计算检索指标。

    :param retrievals: 查询 ID → 检索结果的有序 item ID 列表（相关度降序）
    :param gold: 查询 ID → 相关 item ID 集合
    :param k: 只看前 k 个结果的窗口
    :returns: ``{"recall@k", "ndcg@k", "mrr", "k", "queries"}``
    """
    if k <= 0:
        raise ValueError(f"k 必须为正整数，收到 {k}")

    scored = [
        (query_id, list(gold[query_id]), list(retrievals.get(query_id, [])))
        for query_id in gold
        if gold.get(query_id)
    ]

    if not scored:
        return {"recall@k": 0.0, "ndcg@k": 0.0, "mrr": 0.0, "k": k, "queries": 0}

    recall_sum = 0.0
    ndcg_sum = 0.0
    mrr_sum = 0.0

    for _query_id, relevant, ranked in scored:
        relevant_set = set(relevant)
        top = ranked[:k]
        hits = [item in relevant_set for item in top]

        recall_sum += (sum(hits) / len(relevant_set)) if relevant_set else 0.0

        ideal_hits = [True] * min(len(relevant_set), k)
        idcg = _dcg(ideal_hits, k)
        ndcg_sum += (_dcg(hits, k) / idcg) if idcg > 0 else 0.0

        reciprocal_rank = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                reciprocal_rank = 1.0 / rank
                break
        mrr_sum += reciprocal_rank

    count = len(scored)
    return {
        "recall@k": recall_sum / count,
        "ndcg@k": ndcg_sum / count,
        "mrr": mrr_sum / count,
        "k": k,
        "queries": count,
    }
