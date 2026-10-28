"""HTTP 契约测试：前端照着这个 JSON 写，所以字段要锁住。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from radtracerag.api.app import create_app
from radtracerag.api.engine import build_demo_engine


def _client() -> TestClient:
    return TestClient(create_app(engine=build_demo_engine()))


def test_健康检查报告存储后端() -> None:
    with _client() as client:
        payload = client.get("/api/health").json()

    assert payload["status"] == "ok"
    assert payload["engine"] == "demo"
    assert payload["vector_storage"] == "MilvusVectorDBStorage"
    assert [m["id"] for m in payload["modes"]] == ["none", "naive", "mix"]


def test_模式列表给出三组对照() -> None:
    with _client() as client:
        modes = client.get("/api/modes").json()

    assert [m["group"] for m in modes] == ["A", "B", "C"]
    assert modes[0]["label"] == "无 RAG"


def test_问答返回完整叙事流() -> None:
    with _client() as client:
        response = client.post(
            "/api/query",
            data={"question": "图中主要器官的作用是什么？", "mode": "mix"},
        )
        payload = response.json()

    assert response.status_code == 200
    assert payload["mode"] == "mix"
    assert payload["mode_group"] == "C"
    assert payload["answer"]["text"]
    assert payload["observation"]["entities"]
    assert payload["evidence"]
    assert payload["trace"][0]["stage"] == "图像理解"
    assert payload["trace"][-1]["stage"] == "基于证据回答"
    assert {"value", "label"} == set(payload["metrics"][0])


def test_A组不给证据() -> None:
    with _client() as client:
        payload = client.post(
            "/api/query",
            data={"question": "图中主要器官的作用是什么？", "mode": "none"},
        ).json()

    assert payload["mode_group"] == "A"
    assert payload["evidence"] == []
    assert payload["answer"]["evidence_ids"] == []


def test_B组给候选证据但只引用命中的() -> None:
    with _client() as client:
        payload = client.post(
            "/api/query",
            data={"question": "图中主要器官的作用是什么？", "mode": "naive"},
        ).json()

    assert payload["mode_group"] == "B"
    # 候选 4 条，但答案只引用命中的两条
    assert len(payload["evidence"]) == 4
    assert payload["answer"]["evidence_ids"] == ["1", "3"]
    assert [s["stage"] for s in payload["trace"]] == [
        "图像理解",
        "查询构造",
        "向量检索",
        "命中",
        "基于证据回答",
    ]


def test_C组多一步图扩展() -> None:
    with _client() as client:
        payload = client.post(
            "/api/query",
            data={"question": "图中主要器官的作用是什么？", "mode": "mix"},
        ).json()

    assert payload["mode_group"] == "C"
    assert "图扩展" in [s["stage"] for s in payload["trace"]]
    assert len(payload["answer"]["evidence_ids"]) == 4


def test_未知模式返回400() -> None:
    with _client() as client:
        response = client.post(
            "/api/query",
            data={"question": "问题", "mode": "不存在的模式"},
        )

    assert response.status_code == 400


def test_空问题返回400() -> None:
    with _client() as client:
        response = client.post("/api/query", data={"question": "   ", "mode": "mix"})

    assert response.status_code == 400


def test_openapi_能生成前端类型() -> None:
    with _client() as client:
        schema = client.get("/openapi.json").json()

    assert "/api/query" in schema["paths"]
    assert "QueryResultOut" in schema["components"]["schemas"]
