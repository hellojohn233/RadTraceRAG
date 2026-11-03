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
    #: 语料目录里发现的受支持文件总数（不管最终入库几个）
    discovered: int = 0


def discover_files(root: Path) -> list[Path]:
    """递归发现受支持的文件，按路径排序返回。

    符号链接怎么处理
    ----------------
    - **文件本身是符号链接** → 跳过。链接可能指向语料目录之外，
      把外部内容索引进知识库就是信息泄漏。
    - **语料根目录的祖先是符号链接** → **正常收录**。macOS 上
      ``/var -> /private/var``、``/tmp -> /private/tmp``，临时目录和
      不少用户主目录都长这样。早期版本用「祖先链里不能出现符号链接」
      兜底，结果整棵语料被静默丢弃 —— 摄入 0 个文件却报告成功，
      比直接报错更糟。
    - 判定「有没有跑出语料根目录」用 **resolve 之后的包含关系**，
      而不是逐级检查符号链接：resolve 会把祖先的符号链接一并规范化，
      根目录和文件同步规范化之后，包含关系才有意义。
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"语料目录不存在: {root}")

    resolved_root = root.resolve()
    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        # 文件本身是链接 → 跳过（可能指向语料外）
        if path.is_symlink():
            continue
        # resolve 后必须仍在语料根目录内
        try:
            resolved = path.resolve()
        except OSError:  # pragma: no cover - 极少见的路径解析失败
            continue
        if not resolved.is_relative_to(resolved_root):
            continue
        found.append(path)
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
        if known is None:
            continue
        # 比的是 resolve 之后的真实路径，不是字符串：同一个文件用相对路径
        # 和绝对路径写出来也不该被判成「两个同名文件」。
        try:
            same_file = Path(known).resolve() == path.resolve()
        except OSError:  # pragma: no cover - 极少见的路径解析失败
            same_file = False
        if not same_file:
            conflicts.add(path.name)

    if conflicts:
        reason = (
            "文件名与已入库文档冲突（LightRAG 按文件名判重，同名文件无法共存）: "
            + ", ".join(sorted(conflicts))
            + "。请给冲突文件改名，或清空索引目录（RAG_WORKING_DIR）后重新摄入。"
        )
        return IngestReport(
            failed=tuple(
                (path.relative_to(root).as_posix(), reason)
                for path in files
                if path.name in conflicts
            ),
            discovered=len(files),
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
        # 存 resolve 后的绝对路径：后续的冲突判定要拿它和别的路径比较，
        # 存相对路径的话，换个工作目录就判不出来了。
        path_map[key] = str(path.resolve())
        inserted.append(relative)

    save_path_map(settings.working_dir, path_map)
    return IngestReport(
        inserted=tuple(inserted),
        skipped=tuple(skipped),
        failed=tuple(failed),
        discovered=len(files),
    )
