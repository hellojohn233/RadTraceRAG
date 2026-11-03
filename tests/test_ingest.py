from pathlib import Path

import pytest

from radtracerag.engine.ingest import (
    build_doc_id,
    discover_files,
    extract_text,
    find_basename_conflicts,
    ingest,
    load_path_map,
    save_path_map,
)


def _make_pdf_with_text(text: str = "Liver handles metabolism") -> bytes:
    """构造一个只含一页、含可抽取文本的最小 PDF。"""
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    content = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET"
    objs[3] = f"<< /Length {len(content)} >>\nstream\n{content}\nendstream"
    out = b"%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n{obj}\nendobj\n".encode()
    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return out


class _FakeRag:
    def __init__(self):
        self.inserted = []

    async def ainsert(self, text, ids=None, file_paths=None, **kwargs):
        self.inserted.append({"text": text, "ids": ids, "file_paths": file_paths})
        return "mock-track-id"


def test_discover_files_only_supported_suffixes(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.txt").write_text("x", encoding="utf-8")
    (tmp_path / "c.pdf").write_bytes(b"%PDF-")
    (tmp_path / "d.rst").write_text("x", encoding="utf-8")
    names = {p.name for p in discover_files(tmp_path)}
    assert names == {"a.md", "b.txt", "c.pdf"}


def test_discover_files_skips_symlinks(tmp_path):
    """符号链接会把语料目录外的内容带进索引，必须跳过。"""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "ok.md").write_text("x", encoding="utf-8")
    outside = tmp_path / "outside.md"
    outside.write_text("不该被索引", encoding="utf-8")
    try:
        (corpus / "link.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台不支持创建符号链接")
    names = {p.name for p in discover_files(corpus)}
    assert names == {"ok.md"}


def test_build_doc_id_is_stable_and_short():
    a = build_doc_id("samples/01_x.md")
    assert a == build_doc_id("samples/01_x.md")
    assert len(a) == 16
    assert a != build_doc_id("samples/02_x.md")


def test_find_basename_conflicts_detects_same_name_in_two_dirs(tmp_path):
    (tmp_path / "d1").mkdir()
    (tmp_path / "d2").mkdir()
    (tmp_path / "d1" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "d2" / "a.md").write_text("y", encoding="utf-8")
    conflicts = find_basename_conflicts(discover_files(tmp_path), tmp_path)
    assert conflicts == ["a.md"]


def test_find_basename_conflicts_empty_when_unique(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "b.md").write_text("y", encoding="utf-8")
    assert find_basename_conflicts(discover_files(tmp_path), tmp_path) == []


def test_path_map_roundtrip(tmp_path):
    save_path_map(tmp_path, {"a.md": "samples/a.md"})
    assert load_path_map(tmp_path) == {"a.md": "samples/a.md"}


def test_load_path_map_returns_empty_when_absent(tmp_path):
    assert load_path_map(tmp_path / "nope") == {}


def test_extract_text_reads_markdown(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# 标题\n正文", encoding="utf-8")
    assert "正文" in extract_text(p)


def test_extract_text_reads_pdf(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(_make_pdf_with_text("Liver handles metabolism"))
    assert "Liver handles metabolism" in extract_text(p)


@pytest.mark.asyncio
async def test_ingest_inserts_and_records_path_map(tmp_path, settings_factory):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("肝负责代谢。", encoding="utf-8")
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    rag = _FakeRag()
    report = await ingest(rag, root, settings)
    assert report.inserted == ("a.md",)
    assert report.failed == ()
    assert rag.inserted
    assert rag.inserted[0]["ids"] == [build_doc_id("a.md")]
    assert load_path_map(settings.working_dir) == {"a.md": str(root / "a.md")}


@pytest.mark.asyncio
async def test_ingest_rejects_basename_conflict(tmp_path, settings_factory):
    root = tmp_path / "corpus"
    (root / "d1").mkdir(parents=True)
    (root / "d2").mkdir()
    (root / "d1" / "a.md").write_text("x", encoding="utf-8")
    (root / "d2" / "a.md").write_text("y", encoding="utf-8")
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "s"))
    report = await ingest(_FakeRag(), root, settings)
    assert report.inserted == ()
    assert report.failed and "a.md" in report.failed[0][0]
    assert len(report.failed) == 2


@pytest.mark.asyncio
async def test_ingest_empty_dir_is_not_a_failure(tmp_path, settings_factory):
    root = tmp_path / "corpus"
    root.mkdir()
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "s"))
    report = await ingest(_FakeRag(), root, settings)
    assert report.inserted == () and report.failed == ()


@pytest.mark.asyncio
async def test_ingest_second_run_skips_already_indexed(tmp_path, settings_factory):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("x", encoding="utf-8")
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "s"))
    first = await ingest(_FakeRag(), root, settings)
    second = await ingest(_FakeRag(), root, settings)
    assert first.inserted == ("a.md",)
    assert second.inserted == ()
    assert second.skipped == ("a.md",)
    assert second.failed == ()


@pytest.mark.asyncio
async def test_ingest_missing_root_raises(tmp_path, settings_factory):
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "s"))
    with pytest.raises(FileNotFoundError):
        await ingest(_FakeRag(), tmp_path / "nope", settings)


@pytest.mark.asyncio
async def test_ingest_rejects_name_conflict_with_already_indexed(tmp_path, settings_factory):
    """LightRAG 按 basename 跨批次判重 —— 同名不同路径必须被拦下，不能静默丢文档。"""
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "s"))
    first_root = tmp_path / "one"
    first_root.mkdir()
    (first_root / "a.md").write_text("x", encoding="utf-8")
    first = await ingest(_FakeRag(), first_root, settings)
    assert first.inserted == ("a.md",)

    second_root = tmp_path / "two"
    second_root.mkdir()
    (second_root / "a.md").write_text("y", encoding="utf-8")
    second = await ingest(_FakeRag(), second_root, settings)
    assert second.inserted == ()
    assert second.failed and second.failed[0][0] == "a.md"
    assert "冲突" in second.failed[0][1]


@pytest.mark.asyncio
async def test_ingest_records_failure_without_stopping(tmp_path, settings_factory):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "good.md").write_text("x", encoding="utf-8")
    (root / "bad.pdf").write_bytes(b"not a pdf")
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "s"))
    report = await ingest(_FakeRag(), root, settings)
    assert report.inserted == ("good.md",)
    assert report.failed and report.failed[0][0] == "bad.pdf"


# --- 符号链接与路径归一化 -------------------------------------------------


def test_discover_files_keeps_files_under_a_symlinked_ancestor(tmp_path):
    """语料根目录的祖先是符号链接时，仍必须找到文件。

    macOS 上 ``/var -> /private/var``、``/tmp -> /private/tmp``，
    临时目录和很多用户主目录都长这样。早期实现用「祖先里不能有符号链接」
    兜底，结果整棵语料被静默丢弃 —— 摄入 0 个文件还报告成功。
    """
    real = tmp_path / "real"
    real.mkdir()
    corpus = real / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("x", encoding="utf-8")
    try:
        linked = tmp_path / "linked"
        linked.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台不支持创建符号链接")

    names = {p.name for p in discover_files(linked / "corpus")}
    assert names == {"a.md"}


def test_discover_files_skips_symlinked_directory_pointing_outside(tmp_path):
    """指向语料目录外的符号链接目录，里面的文件不能进索引。"""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "ok.md").write_text("x", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("不该被索引", encoding="utf-8")
    try:
        (corpus / "linkdir").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("当前平台不支持创建符号链接")

    names = {p.name for p in discover_files(corpus)}
    assert names == {"ok.md"}


@pytest.mark.asyncio
async def test_ingest_same_corpus_by_different_path_spellings_is_skipped(
    tmp_path, monkeypatch, settings_factory
):
    """同一份语料用相对路径和绝对路径两次摄入，不该被判成「文件名冲突」。

    path_map 存的是路径字符串，直接比字符串的话，同一个文件会因为写法
    不同被当成两个不同文件 —— 但 LightRAG 那边看到的是同一个 basename。
    """
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("内容", encoding="utf-8")
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))

    first = await ingest(_FakeRag(), corpus, settings)
    assert first.inserted == ("a.md",)

    monkeypatch.chdir(tmp_path)
    second = await ingest(_FakeRag(), Path("corpus"), settings)
    assert second.inserted == ()
    assert second.skipped == ("a.md",)
    assert second.failed == ()


@pytest.mark.asyncio
async def test_ingest_report_exposes_discovered_count(tmp_path, settings_factory):
    """报告里要能看到发现了几个受支持文件 —— 「摄入 0 个」必须看得见。"""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.md").write_text("x", encoding="utf-8")
    (root / "b.txt").write_text("y", encoding="utf-8")
    (root / "c.rst").write_text("忽略", encoding="utf-8")
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "s"))
    report = await ingest(_FakeRag(), root, settings)
    assert report.discovered == 2
