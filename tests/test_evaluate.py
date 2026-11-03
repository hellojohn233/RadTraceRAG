"""检索指标的自检测试。

用**内置样本**验证指标算得对 —— 不是评测结果，是算法自检。
设计文档 §7 明确要求本轮只做骨架 + 自检，不产出对外指标数字。
"""

import pytest

from radtracerag.engine.evaluate import evaluate


def test_perfect_ranking_scores_one():
    result = evaluate(
        retrievals={"q1": ["a", "b", "c"]},
        gold={"q1": ["a", "b"]},
        k=3,
    )
    assert result["recall@k"] == pytest.approx(1.0)
    assert result["ndcg@k"] == pytest.approx(1.0)
    assert result["mrr"] == pytest.approx(1.0)
    assert result["queries"] == 1


def test_no_relevant_retrieved_scores_zero():
    result = evaluate(
        retrievals={"q1": ["x", "y", "z"]},
        gold={"q1": ["a"]},
        k=3,
    )
    assert result["recall@k"] == pytest.approx(0.0)
    assert result["ndcg@k"] == pytest.approx(0.0)
    assert result["mrr"] == pytest.approx(0.0)


def test_partial_recall_is_proportion():
    result = evaluate(
        retrievals={"q1": ["a", "x"]},
        gold={"q1": ["a", "b"]},
        k=2,
    )
    assert result["recall@k"] == pytest.approx(0.5)


def test_missing_query_counts_as_zero():
    """gold 里有、retrievals 里没有的查询算 0 分，不能被悄悄丢掉。"""
    result = evaluate(
        retrievals={"q1": ["a"]},
        gold={"q1": ["a"], "q2": ["b"]},
        k=1,
    )
    assert result["queries"] == 2
    assert result["recall@k"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)


def test_relevant_beyond_window_is_not_counted():
    """k 之外的相关结果不计入 —— 窗口的意义就在这里。"""
    result = evaluate(
        retrievals={"q1": ["x", "y", "a"]},
        gold={"q1": ["a"]},
        k=2,
    )
    assert result["recall@k"] == pytest.approx(0.0)
    assert result["mrr"] == pytest.approx(0.0)


def test_reciprocal_rank_uses_first_hit_position():
    result = evaluate(
        retrievals={"q1": ["x", "a", "b"]},
        gold={"q1": ["a", "b"]},
        k=3,
    )
    assert result["mrr"] == pytest.approx(0.5)


def test_ndcg_penalises_lower_ranked_hit():
    """命中的位置越靠后，nDCG 越低 —— 这是它比召回率多出来的信息。"""
    high = evaluate(retrievals={"q1": ["a", "x"]}, gold={"q1": ["a"]}, k=2)
    low = evaluate(retrievals={"q1": ["x", "a"]}, gold={"q1": ["a"]}, k=2)
    assert high["ndcg@k"] > low["ndcg@k"]
    assert high["recall@k"] == low["recall@k"]


def test_empty_gold_is_not_scored():
    result = evaluate(
        retrievals={"q1": ["a"]},
        gold={"q1": []},
        k=1,
    )
    assert result["queries"] == 0
    assert result["recall@k"] == 0.0


def test_non_positive_k_is_rejected():
    with pytest.raises(ValueError):
        evaluate(retrievals={"q1": ["a"]}, gold={"q1": ["a"]}, k=0)
