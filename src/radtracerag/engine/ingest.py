"""语料摄入：发现文件 → 抽取文本 → 写入 LightRAG 索引 → 维护路径映射。

两件 LightRAG 不替我们做的事
------------------------------
1. **PDF 解析**：传 ``ids`` 进 ``ainsert`` 一定走 RAW 路径
   （``pipeline.py`` 里明确写着「ids outranks docs_format by design」），
   LightRAG 内置的 pypdf 解析只在服务端上传路径生效。所以 PDF 自己抽。
2. **完整路径**：LightRAG 会把 ``file_path`` 截成 basename，引用里因此
   看不到目录。我们自己维护 ``path_map.json``（basename → 完整路径），
   检索时用它还原。

幂等
----
同一个文件重复摄入算「跳过」，不算失败 —— 所以 ``ingest`` 连跑两次的
退出码仍然是 0。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from radtracerag.engine.settings import EngineSettings

#: 支持摄入的文本类后缀
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

#: 路径映射文件名（存放在工作目录下）
PATH_MAP_FILENAME = "path_map.json"


@dataclass(frozen=True)
class IngestReport:
    """一次摄入的结果。"""

    inserted: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    #: (相对路径, 失败原因) 序列
    failed: tuple[tuple[str, str], ...] = ()


def discover_files(root: Path) -> list[Path]:
    """递归发现受支持的文件，按路径排序返回。"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"语料目录不存在: {root}")
    found = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        # 跳过符号链接：指向目录外的链接会把外部内容带进索引
        and not path.is_symlink()
        # 目录符号链接下钻也会跑到语料之外
        and not any(parent.is_symlink() for parent in path.parents)
    ]
    return sorted(found)


def build_doc_id(relative_path: str) -> str:
    """由相对路径生成稳定 doc_id。

    用相对路径而不是绝对路径，这样同一个语料目录在不同机器上生成的
    doc_id 一致，索引也可以跨机器搬运。
    """
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]


def extract_text(path: Path) -> str:
    """抽取文件纯文本。PDF 走 pypdf，其余按 UTF-8 直读。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(str(path))
        except Exception as exc:  # 损坏的 PDF
            raise ValueError(f"PDF 解析失败: {exc}") from exc
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)
    return path.read_text(encoding="utf-8", errors="replace")


def load_path_map(working_dir: Path) -> dict[str, str]:
    """读取路径映射；文件不存在时返回空字典。"""
    path = Path(working_dir) / PATH_MAP_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def save_path_map(working_dir: Path, mapping: dict[str, str]) -> None:
    """写出路径映射（原子替换，避免写一半被读到）。"""
    directory = Path(working_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / PATH_MAP_FILENAME
    temp = target.with_suffix(".json.tmp")
    temp.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(target)


def find_basename_conflicts(paths: list[Path], root: Path) -> list[str]:
    """找出在多个目录下出现同名的文件。

    LightRAG 按 basename 归并引用，同名文件会互相覆盖，所以必须先拦下来。
    """
    root = Path(root)
    seen: dict[str, set[str]] = {}
    for path in paths:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.as_posix()
        seen.setdefault(path.name, set()).add(relative)
    return sorted(name for name, relatives in seen.items() if len(relatives) > 1)


async def ingest(
    rag, root: Path, settings: EngineSettings
) -> IngestReport:
    """把 ``root`` 下的所有受支持文件写入索引。

    - 同名文件（不同目录）→ 全部拒绝，计入 ``failed``；
    - 已入库的文件 → 计入 ``skipped``；
    - 单个文件抽取失败 → 计入 ``failed``，不影响其他文件。
    """
    root = Path(root)
    files = discover_files(root)
    path_map = load_path_map(settings.working_dir)

    # LightRAG 按文件名（basename）判重，且这个判重索引跨批次、跨语料目录生效。
    # 同一 basename 指向不同文件时，后一个不会被索引 —— 所以必须在这里拦下来，
    # 不能让 ainsert 静默丢掉文档。
    conflicts = set(find_basename_conflicts(files, root))
    for path in files:
        known = path_map.get(path.name)
        if known is not None and known != str(path):
            conflicts.add(path.name)

    if conflicts:
        reason = (
            "文件名与已入库文档冲突（LightRAG 按文件名判重，同名文件无法共存）: "
            + ", ".join(sorted(conflicts))
        )
        return IngestReport(
            failed=tuple(
                (path.relative_to(root).as_posix(), reason)
                for path in files
                if path.name in conflicts
            ),
        )

    inserted: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for path in files:
        relative = path.relative_to(root).as_posix()
        key = path.name
        if key in path_map:
            skipped.append(relative)
            continue
        try:
            text = extract_text(path)
            await rag.ainsert(
                text,
                ids=[build_doc_id(relative)],
                file_paths=[str(path)],
            )
        except Exception as exc:
            failed.append((relative, f"{type(exc).__name__}: {exc}"))
            continue
        path_map[key] = str(path)
        inserted.append(relative)

    save_path_map(settings.working_dir, path_map)
    return IngestReport(
        inserted=tuple(inserted),
        skipped=tuple(skipped),
        failed=tuple(failed),
    )
