from pathlib import Path

from radtracerag.engine.contracts import Answer, Document, Evidence, RetrievalStep


def test_evidence_has_no_score_field():
    assert "score" not in Evidence.__dataclass_fields__


def test_evidence_fields():
    e = Evidence(
        evidence_id="E1",
        chunk_id="c1",
        reference_id="1",
        content="内容",
        source="samples/a.md",
    )
    assert e.evidence_id == "E1"
    assert e.source == "samples/a.md"


def test_answer_defaults():
    a = Answer(text="hi")
    assert a.evidence_ids == () and a.unsupported_markers == ()


def test_retrieval_step_defaults():
    s = RetrievalStep(stage="检索", detail="mix")
    assert s.elapsed_ms == 0 and s.count == 0


def test_document_fields():
    d = Document(path=Path("a.md"), doc_id="x", media_type="text/markdown")
    assert d.doc_id == "x"
    assert d.media_type == "text/markdown"
