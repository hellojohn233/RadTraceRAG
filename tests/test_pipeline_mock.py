"""无 API key 的端到端验证。

假 LLM + 假 embedding 驱动真实的 LightRAG，跑通
「ingest → retrieve → answer」整条链路，并验证几个只有真跑才能发现的行为：

- LightRAG 会把 ``file_path`` 截成 basename，完整路径靠我们自己的 path_map 还原；
- ``reference_id`` 是文件级的，同一文件的多个 chunk 共用一个编号；
- 证据编号 ``E1/E2...`` 由本项目生成，与 LightRAG 的 ``reference_id`` 分开；
- 重复 ingest 计入 skipped 而非 failed。
"""

import hashlib
import json

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc

from radtracerag.engine.answer import answer_question
from radtracerag.engine.ingest import ingest, load_path_map
from radtracerag.engine.rag import rag_session
from radtracerag.engine.retrieve import retrieve

DIM = 64


async def _fake_embed(texts, max_token_size=None):
    """假 embedding：必须是 async，且维度严格等于 embedding_dim。"""
    out = []
    for text in texts:
        raw = (
            np.frombuffer(hashlib.sha256(text.encode()).digest(), dtype=np.uint8)
            .astype(np.float32)
            / 255.0
        )
        out.append(np.resize(raw, DIM))
    return np.vstack(out)


async def _fake_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    """假 LLM：按 system_prompt 的角色返回对应形状的 JSON。"""
    text = (system_prompt or "").lower()
    if "knowledge graph specialist" in text:
        return json.dumps({"entities": [], "relationships": []}, ensure_ascii=False)
    if "keyword" in text:
        return json.dumps(
            {"high_level_keywords": ["liver"], "low_level_keywords": ["metabolism"]},
            ensure_ascii=False,
        )
    return "{}"


@pytest.fixture
def mock_models(monkeypatch):
    """替换 LightRAG 侧的 LLM 与 embedding，但保留真实的检索与索引逻辑。"""
    monkeypatch.setattr(
        "radtracerag.engine.rag.build_embedding_func",
        lambda settings: EmbeddingFunc(embedding_dim=DIM, func=_fake_embed, model_name="mock-emb"),
    )
    monkeypatch.setattr(
        "radtracerag.engine.rag.build_llm_func",
        lambda settings: _fake_llm,
    )


@pytest.fixture
def corpus(tmp_path):
    """每个测试用**唯一的文件名与内容**。

    LightRAG 按 basename 和 content hash 判重，且该判重索引在进程内跨实例生效 ——
    若沿用固定文件名，第二个测试的文档会被判重而根本不入库。
    """
    root = tmp_path / "corpus"
    root.mkdir()
    unique = tmp_path.name
    for name in (
        "01_abdominal_anatomy.md",
        "02_liver_function.md",
        "03_ct_imaging_basics.md",
        "04_spleen_pancreas.md",
    ):
        filename = f"{unique}_{name}"
        (root / filename).write_text(
            f"# {filename}\n\n这是 {filename} 的演示内容。肝脏负责代谢和解毒，合成胆汁。\n",
            encoding="utf-8",
        )
    return root


@pytest.mark.asyncio
async def test_pipeline_end_to_end_with_mock(tmp_path, settings_factory, mock_models, corpus, monkeypatch):
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))

    async with rag_session(settings) as rag:
        report = await ingest(rag, corpus, settings)
        assert len(report.inserted) == 4
        assert report.failed == ()

        path_map = load_path_map(settings.working_dir)
        # key 是 basename，value 记录完整路径（项目外 → 绝对路径）
        assert len(path_map) == 4
        assert all(str(corpus) in value.path for value in path_map.values())
        liver_key = next(k for k in path_map if k.endswith("02_liver_function.md"))
        assert path_map[liver_key].path == str((corpus / liver_key).resolve())

        result = await retrieve(rag, "肝脏做什么", "naive", settings, path_map)

    assert result.evidences, "检索应有命中"
    # 完整路径来自我们自己的映射表，而不是 LightRAG 截断后的 basename
    for evidence in result.evidences:
        assert evidence.source.startswith(str(corpus))
    assert result.evidences[0].evidence_id == "E1"
    assert result.evidences[0].chunk_id
    assert result.evidences[0].reference_id


@pytest.mark.asyncio
async def test_answer_uses_evidence_and_drops_fabricated_markers(
    tmp_path, settings_factory, mock_models, corpus, monkeypatch
):
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))

    async with rag_session(settings) as rag:
        report = await ingest(rag, corpus, settings)
        path_map = load_path_map(settings.working_dir)
        result = await retrieve(rag, "肝脏做什么", "naive", settings, path_map)

    async def _fake_complete(prompt, system_prompt=None, settings=None):
        return "肝脏负责代谢与解毒 [E1]。另有说法 [E99]。"

    monkeypatch.setattr("radtracerag.engine.answer._complete", _fake_complete)
    answer = await answer_question("肝脏做什么", result.evidences, settings)

    assert "[E1]" in answer.text
    assert "[E99]" not in answer.text
    assert answer.unsupported_markers == ("E99",)
    assert answer.evidence_ids == ("E1",)


@pytest.mark.asyncio
async def test_second_ingest_skips_everything(
    tmp_path, settings_factory, mock_models, corpus
):
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    async with rag_session(settings) as rag:
        first = await ingest(rag, corpus, settings)
        second = await ingest(rag, corpus, settings)
    assert len(first.inserted) == 4
    assert second.inserted == ()
    assert len(second.skipped) == 4
    assert second.failed == ()


@pytest.mark.asyncio
async def test_content_duplicate_is_reported_as_skipped_not_inserted(
    tmp_path, settings_factory, mock_models
):
    """内容相同、文件名不同的两份文档，LightRAG 只收一份。

    ``ainsert`` **不抛异常** —— 拒收的结果只写在 ``doc_status`` 里。
    不去问 doc_status，就会把没进索引的文档报成 inserted：用户看到
    「新增 2」，实际索引里只有 1 份。
    """
    root = tmp_path / "dup"
    root.mkdir()
    unique = tmp_path.name
    body = "# 重复内容\n\n这两份文件的内容完全一样，只有文件名不同。\n"
    (root / f"{unique}_x.md").write_text(body, encoding="utf-8")
    (root / f"{unique}_y.md").write_text(body, encoding="utf-8")

    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    async with rag_session(settings) as rag:
        report = await ingest(rag, root, settings)

    assert len(report.inserted) == 1, "只有一份能真正进索引"
    assert len(report.duplicates) == 1, "被判重的那份应单独报出来，不是失败"
    assert report.skipped == ()
    assert report.failed == ()


@pytest.mark.asyncio
async def test_ingest_after_path_map_loss_reports_duplicates_not_inserted(
    tmp_path, settings_factory, mock_models, corpus
):
    """path_map 丢了之后重跑，不能再报「已入库」。

    上游会按 doc_id 判重：同一份文档第二次插入时，它另起一条 ``dup-*``
    记录并把这次插入标记为失败。而我们这个 doc_id 在 doc_status 里**还留着
    上一轮的 processed 行** —— 直接查它会把「本次被拒收」误判成「已入库」。

    判据只能是**本次 track_id** 下的记录。
    """
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    async with rag_session(settings) as rag:
        first = await ingest(rag, corpus, settings)
        assert len(first.inserted) == 4

        # 模拟 path_map 丢失（文件被删或损坏）
        (settings.working_dir / "path_map.json").unlink()

        second = await ingest(rag, corpus, settings)

    assert second.inserted == (), "上游判重了，本次没有任何新文档入库"
    assert len(second.duplicates) == 4
    assert second.failed == ()


@pytest.mark.asyncio
async def test_ingest_restores_path_map_after_duplicates(
    tmp_path, settings_factory, mock_models, corpus
):
    """被判重的文件仍然要把 path_map 写回去 —— 否则每次重跑都要白跑一遍上游。"""
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    async with rag_session(settings) as rag:
        await ingest(rag, corpus, settings)
        (settings.working_dir / "path_map.json").unlink()
        second = await ingest(rag, corpus, settings)

    assert len(second.duplicates) == 4
    assert len(load_path_map(settings.working_dir)) == 4
