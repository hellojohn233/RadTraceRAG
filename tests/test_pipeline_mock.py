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
import shutil

import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc

from radtracerag.engine.answer import answer_question
from radtracerag.engine.ingest import build_doc_id, ingest, load_path_map
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
async def test_ingest_after_path_map_loss_reports_already_indexed_not_inserted(
    tmp_path, settings_factory, mock_models, corpus
):
    """path_map 丢了之后重跑，不能再报「新增」。

    上游会按 doc_id 判重：同一份文档第二次插入时，它另起一条 ``dup-*``
    记录并把这次插入标记为失败。而我们这个 doc_id 在 doc_status 里**还留着
    上一轮的 processed 行** —— 直接查它会把「本次被拒收」误判成「已入库」。

    判据只能是**本次 track_id** 下的记录。

    报告的口径：上游报的是「文件名重复」，而原文档就是我们自己的 doc_id
    —— 这几份文件确实在索引里（第一轮真插进去了），所以是「已入库」，
    不是「内容与别人重复」。
    """
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    async with rag_session(settings) as rag:
        first = await ingest(rag, corpus, settings)
        assert len(first.inserted) == 4

        # 模拟 path_map 丢失（文件被删或损坏）
        (settings.working_dir / "path_map.json").unlink()

        second = await ingest(rag, corpus, settings)

    assert second.inserted == (), "上游判重了，本次没有任何新文档入库"
    assert len(second.skipped) == 4
    assert {reason for _name, reason in second.skipped} == {"已入库"}
    assert second.duplicates == ()
    assert second.failed == ()


@pytest.mark.asyncio
async def test_failed_document_is_reported_as_failed_on_retry(
    tmp_path, settings_factory, mock_models, corpus, monkeypatch
):
    """上一轮上游失败的文档，本轮**不能**被报成「已在索引中」。

    上游一旦给某个 doc_id 记过 doc_status，就**永久拒收**它的重插 ——
    所以失败的文档下一轮会被上游判成「文件名重复」。这时候如果顺着
    duplicate_kind 就报「已入库」，等于把一个从没进过索引的文件报告成
    入库了，而且这个误报还会被写进 path_map 永久生效。

    正确做法：回 doc_status 看**我们自己那行记录的真实状态** ——
    它是 failed，就报失败，并把上游的原始错误带出来。
    """
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))

    async def _boom(*args, **kwargs):
        raise RuntimeError("simulated upstream LLM failure")

    monkeypatch.setattr("radtracerag.engine.rag.build_llm_func", lambda s: _boom)

    async with rag_session(settings) as rag:
        first = await ingest(rag, corpus, settings)
        assert first.inserted == ()
        assert first.failed, "上游故障必须报成失败"

    # 恢复正常
    monkeypatch.setattr("radtracerag.engine.rag.build_llm_func", lambda s: _fake_llm)

    async with rag_session(settings) as rag:
        second = await ingest(rag, corpus, settings)

    assert second.inserted == (), "上游不会重新处理失败的 doc_id"
    assert second.duplicates == (), "不能谎报成「已在索引中」"
    assert second.failed, "必须报成失败"
    assert "simulated upstream LLM failure" in second.failed[0][1]
    # 失败的文档要**记住 doc_id**：下一轮直接核实上游状态，不再重复提交。
    # 否则上游每轮多一条失败记录，doc_status.json 无限膨胀。
    stored = load_path_map(settings.working_dir)
    assert stored, "失败的文档也要记住 doc_id"
    assert all(entry.doc_id for entry in stored.values())

    async with rag_session(settings) as rag:
        before = _count_doc_status_rows(settings)
        third = await ingest(rag, corpus, settings)
        after = _count_doc_status_rows(settings)

    assert third.failed, "第三轮仍然要报失败"
    assert after == before, "失败的文档不能每轮都再提交一次（上游失败记录会无限增长）"


@pytest.mark.asyncio
async def test_content_changed_after_path_map_loss_is_reported_as_stale(
    tmp_path, settings_factory, mock_models, corpus
):
    """path_map 丢了 + 文件内容改过 → 必须说「索引里仍是旧版本」。

    上游 doc_status 的主行带着 ``content_hash``（上游口径：正文先过
    ``sanitize_text_for_encoding`` 再取 md5）—— 拿它和当前文件比一比就知道
    索引里躺着的是哪一版。

    而且这个提醒**不能只出现一次**：上一次实现把当前文件的哈希写回了
    path_map，下一次就以为「已入库」，过期状态被自己抹掉了。
    """
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    async with rag_session(settings) as rag:
        first = await ingest(rag, corpus, settings)
        assert len(first.inserted) == 4

        (settings.working_dir / "path_map.json").unlink()
        target = corpus / sorted(p.name for p in corpus.iterdir())[0]
        target.write_text("完全不同的新内容。\n", encoding="utf-8")

        second = await ingest(rag, corpus, settings)
        third = await ingest(rag, corpus, settings)

    assert second.inserted == ()
    stale_reasons = [
        reason for _name, reason in second.skipped if "内容已变化" in reason
    ]
    assert stale_reasons, f"改过内容的文件必须报「内容已变化」：{second.skipped}"
    third_reasons = [
        reason for _name, reason in third.skipped if "内容已变化" in reason
    ]
    assert third_reasons, "过期提醒不能只出现一次"
    stored = load_path_map(settings.working_dir)
    assert any(entry.stale for entry in stored.values()), "过期状态要落到 path_map 里"



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

    assert len(second.skipped) == 4
    assert len(load_path_map(settings.working_dir)) == 4


def _count_doc_status_rows(settings, basenames=None) -> int:
    """数上游 doc_status 里有多少行（含 dup-* 记录）。

    每轮 ingest 都多几行 → 说明我们在重复提交注定失败的文档。

    ⚠️ ``basenames`` 限定统计范围：**上游的 doc_status 存储是进程级共享
    命名空间** —— 同一个进程里所有索引实例的记录都落在同一张表上，
    拿整张表做断言会串到别的测试（第一次写盘时会把别人的行一起写进去）。
    """
    path = settings.working_dir / "kv_store_doc_status.json"
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    if basenames is None:
        return len(data)
    wanted = set(basenames)
    return sum(
        1
        for row in data.values()
        if isinstance(row, dict) and str(row.get("file_path", "")) in wanted
    )


@pytest.mark.asyncio
async def test_document_deleted_from_index_is_reingested(
    tmp_path, settings_factory, mock_models, corpus
):
    """索引里的文档被外部删掉后，重跑 ingest 必须把它补回去。

    path_map 还在、文档却不在索引里了 —— 只信 path_map 就会一直报
    「已入库」，而这份文档**永远补不回来**（每轮都跳过）。
    这与刚修掉的 C-1 是同一族失败：谎报「已入库」= 静默丢文档。
    """
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    async with rag_session(settings) as rag:
        first = await ingest(rag, corpus, settings)
        assert len(first.inserted) == 4

        target = sorted(corpus.iterdir())[0]
        doc_id = build_doc_id(target.name)
        await rag.adelete_by_doc_id(doc_id)

        second = await ingest(rag, corpus, settings)

    assert second.inserted == (target.name,), "索引里没有的文档必须重新入库"
    assert second.failed == ()


@pytest.mark.asyncio
async def test_moved_file_with_lost_path_map_is_not_a_failure(
    tmp_path, settings_factory, mock_models
):
    """文件在语料目录里搬过家 + path_map 丢失 → 不能报失败。

    相对路径变了 → 我们的 doc_id 变了；basename 没变 → 上游按文件名判重，
    把我们拒了。但**内容其实在索引里**（挂在原 doc_id 下）。
    不查原文档那行的 content_hash 就报失败的话：退出码变 1（破坏
    「ingest 幂等」的承诺），而且因为不写 path_map，每一轮都失败。
    """
    work = tmp_path / "work"
    corpus = work / "corpus"
    (corpus / "sub").mkdir(parents=True)
    settings = settings_factory(RAG_WORKING_DIR=str(work / "storage"))
    body = "搬过家的内容。\n"
    (corpus / "sub" / "moved.md").write_text(body, encoding="utf-8")

    async with rag_session(settings) as rag:
        first = await ingest(rag, corpus, settings)
        assert first.inserted == ("sub/moved.md",)

        shutil.move(str(corpus / "sub" / "moved.md"), str(corpus / "moved.md"))
        (settings.working_dir / "path_map.json").unlink()

        second = await ingest(rag, corpus, settings)

    assert second.failed == (), "内容在索引里，不该报失败"
    assert any("已入库" in reason for _name, reason in second.skipped), second.skipped


@pytest.mark.asyncio
async def test_moved_files_do_not_grow_upstream_records(
    tmp_path, settings_factory, mock_models
):
    """文件搬过家之后反复 ingest，上游的记录数**不能增长**。

    踩过的坑：核实「这份文档在不在索引里」时，用的是**当前相对路径**算出来的
    doc_id —— 而 path_map 的 key 是 basename（与位置无关）。文件在语料目录里
    搬过家 → 相对路径变了 → doc_id 变了 → 查上游必然查不到 → 误判成
    「不在索引里」→ 每轮重新提交 → 上游每轮多一条 dup-* 失败记录，
    doc_status.json 无限膨胀（实测 18→32→46→60→74 KB）。

    修法：path_map 记住「这份内容在索引里挂在哪个 doc_id 下」。
    """
    work = tmp_path / "work"
    corpus = work / "corpus"
    (corpus / "sub").mkdir(parents=True)
    settings = settings_factory(RAG_WORKING_DIR=str(work / "storage"))
    for index in range(3):
        (corpus / "sub" / f"moved_{index}.md").write_text(
            f"文件 {index} 的内容。\n", encoding="utf-8"
        )

    async with rag_session(settings) as rag:
        first = await ingest(rag, corpus, settings)
        assert len(first.inserted) == 3
        rows_after_first = _count_doc_status_rows(settings)

    # 搬到根目录
    for index in range(3):
        shutil.move(
            str(corpus / "sub" / f"moved_{index}.md"),
            str(corpus / f"moved_{index}.md"),
        )

    for _round in range(3):
        async with rag_session(settings) as rag:
            report = await ingest(rag, corpus, settings)
        assert report.failed == ()
        assert report.inserted == ()
        rows = _count_doc_status_rows(settings)
        assert rows == rows_after_first, (
            f"上游记录数不该增长：{rows_after_first} → {rows}"
        )


@pytest.mark.asyncio
async def test_upstream_row_not_processed_is_reported_as_failure(
    tmp_path, settings_factory, mock_models
):
    """上游那一行存在但状态不是 processed → 报失败，不能报「已入库」。

    上游有这一行、状态是 pending → 文档**不在索引里**（没处理完），
    报成「已入库」就是谎报。

    ⚠️ 改状态必须用**内存 API**：上游的 doc_status 是进程级共享命名空间，
    ``initialize()`` 只在首次读磁盘文件 —— 同进程内改文件没有任何效果。
    """
    work = tmp_path / "work"
    corpus = work / "corpus"
    corpus.mkdir(parents=True)
    settings = settings_factory(RAG_WORKING_DIR=str(work / "storage"))
    (corpus / "pending.md").write_text("内容。\n", encoding="utf-8")

    async with rag_session(settings) as rag:
        first = await ingest(rag, corpus, settings)
        assert first.inserted == ("pending.md",)

        doc_id = build_doc_id("pending.md")
        row = await rag.doc_status.get_by_id(doc_id)
        row["status"] = "DocStatus.PENDING"
        await rag.doc_status.upsert({doc_id: row})

        second = await ingest(rag, corpus, settings)

    assert second.inserted == ()
    assert second.failed, "上游状态不是 processed，必须报失败"
    assert not any("已入库" in reason for _name, reason in second.skipped)


@pytest.mark.asyncio
async def test_empty_document_does_not_grow_doc_status_every_round(
    tmp_path, settings_factory, mock_models
):
    """空文档每轮重提 → 上游 doc_status 每轮多一行。

    上游的 doc_status 过滤**只看键在不在、不看状态** → 空内容占位那个
    doc_id 永久拒收重插 → 重提永远不会被接受，只是每轮在上游多留一条
    记录。而这张表每轮还要被 O(n) 全表扫描 → 每轮 ingest 都更慢。

    认出来（``_verify_indexed`` 的 empty 态）+ 记住 doc_id → 只提交一次。
    """
    unique = tmp_path.name
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / f"{unique}_blank.md").write_text("   \n\n", encoding="utf-8")
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))

    name = f"{unique}_blank.md"

    async with rag_session(settings) as rag:
        rows_before = _count_doc_status_rows(settings, {name})
        first = await ingest(rag, corpus, settings)
        rows_after_first = _count_doc_status_rows(settings, {name})
        second = await ingest(rag, corpus, settings)
        rows_after_second = _count_doc_status_rows(settings, {name})
        third = await ingest(rag, corpus, settings)
        rows_after_third = _count_doc_status_rows(settings, {name})

    assert len(first.empty) == 1, "空白文件是空占位，不算入库"
    assert len(second.empty) == 1
    assert len(third.empty) == 1
    assert rows_after_first - rows_before == 1, "只提交过一次"
    assert rows_after_second == rows_after_first, "空文档不能每轮重提一次"
    assert rows_after_third == rows_after_first


@pytest.mark.asyncio
async def test_filename_occupied_by_another_document_does_not_grow_doc_status(
    tmp_path, settings_factory, mock_models
):
    """文件名被索引里的另一份文档占着 → 每轮重提 → 上游每轮多一条 dup 记录。

    上游按文件名判重，被拒的提交会留下一条 ``dup-*`` 记录，而它的 id 里
    含 track_id → 每轮重提必然多一条新记录。

    现在提交**之前**先问上游「这个文件名有主吗」→ 直接报失败，不提交 →
    上游一条记录都不多。
    """
    unique = tmp_path.name
    name = f"{unique}_taken.md"
    corpus_a = tmp_path / "a"
    corpus_b = tmp_path / "b"
    corpus_a.mkdir()
    (corpus_b / "sub").mkdir(parents=True)
    (corpus_a / name).write_text("甲文档的正文，讲的是肝脏代谢。", encoding="utf-8")
    # 同一个 basename、不同的相对路径 → 上游按文件名判重会拒掉第二份
    (corpus_b / "sub" / name).write_text(
        "乙文档的正文，讲的是脾脏与胰腺。", encoding="utf-8"
    )
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))

    async with rag_session(settings) as rag:
        first = await ingest(rag, corpus_a, settings)
        assert len(first.inserted) == 1
        # path_map 丢失 → 逼我们走「靠上游判断」那条路
        (settings.working_dir / "path_map.json").unlink()
        rows_before = _count_doc_status_rows(settings, {name})

        second = await ingest(rag, corpus_b, settings)
        rows_after = _count_doc_status_rows(settings, {name})
        third = await ingest(rag, corpus_b, settings)
        rows_after_third = _count_doc_status_rows(settings, {name})

    assert second.failed, "文件名被占用必须报失败（内容根本没进索引）"
    assert rows_after == rows_before, "被拒的文件不能每轮重提一次（上游每轮多一条记录）"
    assert rows_after_third == rows_before


@pytest.mark.asyncio
async def test_duplicate_content_is_detected_before_submitting(
    tmp_path, settings_factory, mock_models
):
    """内容与另一份文档相同 → 上游按内容判重。

    判重同样会留下一条 ``dup-*`` 记录（id 含 track_id → 每轮重提多一条）。

    提交前先问上游「这份内容是不是已经有主了」→ 直接报「内容已在索引中」，
    不提交 → 上游一条记录都不多。
    """
    unique = tmp_path.name
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    body = f"{unique} 的重复正文，讲的是胆汁与代谢。\n"
    (corpus / f"{unique}_first.md").write_text(body, encoding="utf-8")
    (corpus / f"{unique}_second.md").write_text(body, encoding="utf-8")
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))

    names = {f"{unique}_first.md", f"{unique}_second.md"}

    async with rag_session(settings) as rag:
        rows_before = _count_doc_status_rows(settings, names)
        first = await ingest(rag, corpus, settings)
        rows_after_first = _count_doc_status_rows(settings, names)
        second = await ingest(rag, corpus, settings)
        rows_after_second = _count_doc_status_rows(settings, names)

    assert len(first.inserted) == 1, "只应有一份真的进索引"
    assert len(first.duplicates) == 1, "另一份是重复内容，不算失败"
    assert first.failed == ()
    assert (
        rows_after_first - rows_before == 1
    ), "内容判重不该留下 dup-* 记录（只有真入库那份才有一行）"
    assert len(second.duplicates) == 1
    assert rows_after_second == rows_after_first
