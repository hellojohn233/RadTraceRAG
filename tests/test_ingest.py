import asyncio
import shutil
from pathlib import Path

import pytest

from radtracerag.engine.rag import rag_session
from radtracerag.engine.ingest import (
    MappedFile,
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
    save_path_map(tmp_path, {"a.md": MappedFile(path="samples/a.md", sha256="abc")})
    loaded = load_path_map(tmp_path)
    assert loaded["a.md"].path == "samples/a.md"
    assert loaded["a.md"].sha256 == "abc"


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
    stored = load_path_map(settings.working_dir)
    assert stored["a.md"].path == str((root / "a.md").resolve())
    assert stored["a.md"].sha256, "要存内容哈希，用来分辨「同一个文件」和「另一个同名文件」"


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


@pytest.mark.asyncio
async def test_ingest_conflict_does_not_drop_non_conflicting_files(tmp_path, settings_factory):
    """有同名冲突时，只拦冲突的那两个文件，其余照常摄入。

    早先的实现一发现冲突就整批中止，没冲突的文件既没入库也没出现在报告里 ——
    4 个文件发现、0 个入库、2 个失败，剩下 2 个凭空消失。
    """
    root = tmp_path / "corpus"
    (root / "d1").mkdir(parents=True)
    (root / "d2").mkdir()
    (root / "d1" / "dup.md").write_text("x", encoding="utf-8")
    (root / "d2" / "dup.md").write_text("y", encoding="utf-8")
    (root / "unique.md").write_text("z", encoding="utf-8")

    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "s"))
    report = await ingest(_FakeRag(), root, settings)

    assert report.inserted == ("unique.md",), "没冲突的文件必须照常入库"
    assert len(report.failed) == 2, "两个冲突文件都要报出来"
    assert {name for name, _ in report.failed} == {"d1/dup.md", "d2/dup.md"}
    assert report.discovered == 3


def test_find_basename_conflicts_detects_parser_hint_variant(tmp_path):
    """LightRAG 会把 ``a.pdf`` 与 ``a.[native-iet].pdf`` 归并成同一个 key。

    我们的冲突检测如果按原始文件名分组，就会漏判这一对 —— 第二个文件
    会被上游静默丢弃，而我们报告说它入库了。
    """
    (tmp_path / "a.pdf").write_bytes(b"%PDF-")
    (tmp_path / "a.[native-iet].pdf").write_bytes(b"%PDF-")
    conflicts = find_basename_conflicts(discover_files(tmp_path), tmp_path)
    assert conflicts == ["a.pdf"], "冲突要按上游的规范名报告"


@pytest.mark.asyncio
async def test_path_map_key_uses_canonical_basename(tmp_path, settings_factory):
    """path_map 的 key 必须与 LightRAG 存的 file_path 一致，否则检索时还原不出路径。"""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.[native-iet].pdf").write_bytes(_make_pdf_with_text("内容"))
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "s"))
    await ingest(_FakeRag(), root, settings)
    stored = load_path_map(settings.working_dir)
    assert stored["a.pdf"].path == str((root / "a.[native-iet].pdf").resolve())


# --- 复核发现的边界情况 -------------------------------------------------


def test_path_map_survives_project_relocation(tmp_path, settings_factory, monkeypatch):
    """整个项目搬到别的目录后，同一份语料不该被判成「同名冲突」。

    path_map 存绝对路径的话，搬家后每个文件的路径都变了 → 全量假冲突，
    用户唯一的选择是清空索引重建。而索引本身是可以搬家的。
    """
    from radtracerag.engine import ingest as ingest_mod

    project = tmp_path / "proj"
    corpus = project / "corpus"
    corpus.mkdir(parents=True)
    (corpus / "a.md").write_text("内容。", encoding="utf-8")

    monkeypatch.setattr(ingest_mod, "PROJECT_ROOT", project)
    settings = settings_factory(RAG_WORKING_DIR=str(project / "storage"))
    report = asyncio.run(ingest(_FakeRag(), corpus, settings))
    assert report.inserted == ("a.md",)
    stored = load_path_map(settings.working_dir)
    assert stored["a.md"].path == "corpus/a.md", "项目内的路径要存相对项目根的形式"
    assert stored["a.md"].sha256

    # 搬到别的目录：相对路径跟着项目一起搬，解析后仍然指向同一个文件
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.move(str(project), str(moved / "proj"))
    monkeypatch.setattr(ingest_mod, "PROJECT_ROOT", moved / "proj")
    # 搬家后重新解析配置：工作目录也跟着项目走
    moved_settings = settings_factory(RAG_WORKING_DIR=str(moved / "proj" / "storage"))
    report2 = asyncio.run(ingest(_FakeRag(), moved / "proj" / "corpus", moved_settings))
    assert report2.failed == (), "搬家后不该有假冲突"
    assert report2.inserted == () and report2.skipped == ("a.md",)


def test_load_path_map_warns_when_corrupted(tmp_path, capsys):
    """损坏的 path_map 不能静默变成空字典 —— 那是「已入库」判断失效的源头。"""
    (tmp_path / "path_map.json").write_text("{ 不是合法 JSON", encoding="utf-8")
    assert load_path_map(tmp_path) == {}
    captured = capsys.readouterr()
    assert "path_map" in captured.out + captured.err


@pytest.mark.asyncio
async def test_ingest_moved_corpus_outside_project_is_skipped_not_conflict(
    tmp_path, settings_factory
):
    """项目外的语料整体搬了家，也不该被判成「同名冲突」。

    文件还是同一个文件（内容一模一样），上游只是按 basename 判重所以进不去
    索引 —— 这是「跳过」，不是冲突。光比路径分辨不出来（路径确实变了），
    要比内容哈希。
    """
    original = tmp_path / "before"
    original.mkdir()
    (original / "a.md").write_text("完全一样的内容。", encoding="utf-8")

    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    first = await ingest(_FakeRag(), original, settings)
    assert first.inserted == ("a.md",)

    moved = tmp_path / "after"
    moved.mkdir()
    shutil.move(str(original / "a.md"), str(moved / "a.md"))

    second = await ingest(_FakeRag(), moved, settings)
    assert second.failed == (), "搬家不是冲突"
    assert second.skipped == ("a.md",)
    # 映射里的位置要更新到新路径
    assert load_path_map(settings.working_dir)["a.md"].path == str((moved / "a.md").resolve())


@pytest.mark.asyncio
async def test_ingest_different_file_with_same_name_is_conflict(tmp_path, settings_factory):
    """内容不同的同名文件仍然是冲突 —— 不能因为放宽了就放行。"""
    first_dir = tmp_path / "one"
    first_dir.mkdir()
    (first_dir / "a.md").write_text("第一份内容。", encoding="utf-8")

    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    await ingest(_FakeRag(), first_dir, settings)

    second_dir = tmp_path / "two"
    second_dir.mkdir()
    (second_dir / "a.md").write_text("完全不同的另一份内容。", encoding="utf-8")

    report = await ingest(_FakeRag(), second_dir, settings)
    assert report.inserted == ()
    assert report.failed and report.failed[0][0] == "a.md"
    assert "冲突" in report.failed[0][1]
