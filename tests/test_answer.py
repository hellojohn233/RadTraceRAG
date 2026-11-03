import pytest

from radtracerag.engine.answer import answer_question
from radtracerag.engine.contracts import Evidence

_EV = [
    Evidence(
        evidence_id="E1",
        chunk_id="c1",
        reference_id="1",
        content="肝负责代谢和解毒。",
        source="samples/liver.md",
    ),
    Evidence(
        evidence_id="E2",
        chunk_id="c2",
        reference_id="2",
        content="胆囊储存胆汁。",
        source="samples/gall.md",
    ),
]


def _fake_complete(reply: str):
    async def _impl(prompt, system_prompt=None, settings=None):
        return reply

    return _impl


@pytest.mark.asyncio
async def test_answer_keeps_valid_markers(monkeypatch, settings):
    monkeypatch.setattr(
        "radtracerag.engine.answer._complete", _fake_complete("肝负责代谢与解毒 [E1]。")
    )
    a = await answer_question("肝做什么", _EV, settings)
    assert a.evidence_ids == ("E1",)
    assert a.unsupported_markers == ()
    assert "[E1]" in a.text


@pytest.mark.asyncio
async def test_answer_drops_unsupported_markers(monkeypatch, settings):
    monkeypatch.setattr(
        "radtracerag.engine.answer._complete", _fake_complete("结论 [E1]，另有 [E9]。")
    )
    a = await answer_question("肝做什么", _EV, settings)
    assert a.unsupported_markers == ("E9",)
    assert "[E9]" not in a.text
    assert a.evidence_ids == ("E1",)


@pytest.mark.asyncio
async def test_answer_with_no_evidence_does_not_call_llm(monkeypatch, settings):
    def _boom(*args, **kwargs):
        raise AssertionError("不应调用 LLM")

    monkeypatch.setattr("radtracerag.engine.answer._complete", _boom)
    a = await answer_question("肝做什么", [], settings)
    assert a.evidence_ids == ()
    assert "证据" in a.text


@pytest.mark.asyncio
async def test_prompt_lists_evidence_with_ids(monkeypatch, settings):
    captured = {}

    async def _capture(prompt, system_prompt=None, settings=None):
        captured["prompt"] = prompt
        captured["system"] = system_prompt
        return "答案 [E1]"

    monkeypatch.setattr("radtracerag.engine.answer._complete", _capture)
    await answer_question("肝做什么", _EV, settings)
    assert "[E1]" in captured["prompt"]
    assert "[E2]" in captured["prompt"]
    assert "肝负责代谢和解毒。" in captured["prompt"]
    assert "肝做什么" in captured["prompt"]
    assert "证据" in captured["system"]
