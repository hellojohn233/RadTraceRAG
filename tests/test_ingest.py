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
