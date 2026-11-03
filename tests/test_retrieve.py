import pytest

from radtracerag.engine.retrieve import evidence_from_payload, retrieve

_PAYLOAD = {
    "status": "success",
    "data": {
        "entities": [],
        "relationships": [],
        "chunks": [
            {
                "reference_id": "1",
                "content": "肝负责代谢。",
                "file_path": "02_liver_function.md",
                "chunk_id": "d1-chunk-000",
            },
            {
                "reference_id": "1",
                "content": "胆囊储存胆汁。",
                "file_path": "02_liver_function.md",
                "chunk_id": "d1-chunk-001",
            },
            {
                "reference_id": "2",
                "content": "CT 原理。",
                "file_path": "03_ct_basics.md",
                "chunk_id": "d2-chunk-000",
            },
        ],
        "references": [
            {"reference_id": "1", "file_path": "02_liver_function.md"},
            {"reference_id": "2", "file_path": "03_ct_basics.md"},
        ],
    },
    "metadata": {
        "query_mode": "mix",
        "keywords": {"high_level": ["liver"], "low_level": ["metabolism"]},
        "processing_info": {"final_chunks_count": 3},
    },
}


class _FakeRag:
    def __init__(self, payload=None):
        self.calls = []
        self.payload = payload if payload is not None else _PAYLOAD

    async def aquery_data(self, query, param):
        self.calls.append((query, param))
        return self.payload


def test_evidence_ids_are_sequential():
    ev = evidence_from_payload(_PAYLOAD, {})
    assert [e.evidence_id for e in ev] == ["E1", "E2", "E3"]


def test_source_uses_path_map_when_available():
    ev = evidence_from_payload(_PAYLOAD, {"02_liver_function.md": "samples/02_liver_function.md"})
    assert ev[0].source == "samples/02_liver_function.md"


def test_source_falls_back_to_basename_when_not_in_map():
    ev = evidence_from_payload(_PAYLOAD, {})
    assert ev[2].source == "03_ct_basics.md"


def test_evidence_keeps_chunk_id_and_reference_id():
    ev = evidence_from_payload(_PAYLOAD, {})
    assert ev[0].chunk_id == "d1-chunk-000"
    assert ev[0].reference_id == "1"
    assert ev[2].reference_id == "2"


def test_evidence_has_no_score():
    ev = evidence_from_payload(_PAYLOAD, {})
    assert "score" not in ev[0].__dataclass_fields__


def test_missing_data_key_returns_empty():
    assert evidence_from_payload({}, {}) == ()
    assert evidence_from_payload({"data": {}}, {}) == ()


@pytest.mark.asyncio
async def test_retrieve_uses_aquery_data_and_disables_rerank(settings):
    rag = _FakeRag()
    res = await retrieve(rag, "肝做什么", "mix", settings, {})
    assert res.mode == "mix"
    assert len(res.evidences) == 3
    query, param = rag.calls[0]
    assert query == "肝做什么"
    assert param.mode == "mix"
    assert param.enable_rerank is False
    assert param.top_k == settings.top_k
    assert param.chunk_top_k == settings.chunk_top_k


@pytest.mark.asyncio
async def test_retrieve_returns_steps_and_metadata(settings):
    res = await retrieve(_FakeRag(), "q", "naive", settings, {})
    assert res.metadata["query_mode"] == "mix"
    assert res.steps
    assert any(step.stage == "检索" for step in res.steps)


@pytest.mark.asyncio
async def test_retrieve_on_empty_payload_returns_no_evidences(settings):
    res = await retrieve(_FakeRag({"status": "success", "data": {}, "metadata": {}}), "q", "naive", settings, {})
    assert res.evidences == ()
