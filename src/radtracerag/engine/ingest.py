"""语料摄入：发现文件 → 抽取文本 → 写入 LightRAG 索引 → 维护路径映射。

三件 LightRAG 不替我们做的事
----------------------------
1. **PDF 解析**：传 ``ids`` 进 ``ainsert`` 一定走 RAW 路径
   （``pipeline.py`` 里明确写着「ids outranks docs_format by design」），
   LightRAG 内置的 pypdf 解析只在服务端上传路径生效。所以 PDF 自己抽。
2. **完整路径**：LightRAG 会把 ``file_path`` 截成 basename，引用里因此
   看不到目录。我们自己维护 ``path_map.json``（basename → 完整路径），
   检索时用它还原。
3. **入库结果**：``ainsert`` **从不因为拒收而抛异常** —— 它把结果写进
   ``doc_status``，返回的只是 track_id。「调用没报错」**不等于**
   「文档被索引了」。所以每次插入后我们都要回 ``doc_status`` 核实。

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

try:  # pragma: no cover - 上游改名时走兜底分支
    from lightrag.utils_pipeline import normalize_document_file_path
except Exception:  # pragma: no cover
    normalize_document_file_path = None


def canonical_basename(path: Path) -> str:
    """文件名在 LightRAG 眼里的规范形式。

    LightRAG 存 ``file_path`` 时会用 ``normalize_document_file_path`` 剥掉
    解析器提示段 —— ``abc.pdf`` 和 ``abc.[native-iet].pdf`` 会归并成同一个 key
    （``utils_pipeline.py:248``）。我们必须用同一个函数算 key，否则：

    - 同名冲突检测会漏判这一对，第二个文件被上游静默丢弃；
    - ``path_map`` 存的是原始文件名，检索时按规范名查不到，
      引用里的来源路径就还原不出来。
    """
    name = Path(path).name
    if normalize_document_file_path is None:
        return name
    try:
        canonical = normalize_document_file_path(name)
    except Exception:  # pragma: no cover - 上游行为异常时的兜底
        return name
    return canonical or name


@dataclass(frozen=True)
class IngestReport:
    """一次摄入的结果。"""

    inserted: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()
    #: (相对路径, 失败原因) 序列
    failed: tuple[tuple[str, str], ...] = ()
    #: 语料目录里发现的受支持文件总数（不管最终入库几个）
    discovered: int = 0


@dataclass(frozen=True)
class InsertOutcome:
    """一次 ``ainsert`` 的真实结果。

    ``status`` 取值：

    - ``inserted`` —— 真的进索引了
    - ``skipped`` —— 被上游判重（内容与已入库文档相同）
    - ``failed`` —— 上游处理失败，``reason`` 里是上游的原始错误信息
    """

    status: str
    reason: str = ""


def _field(record, name: str, default=None):
    """取记录里的一个字段，兼容 dict 与 dataclass 两种形态。

    上游两种 API 的返回类型不一样：``get_by_id`` 给 dict，
    ``get_docs_by_track_id`` 给 ``DocProcessingStatus`` 数据类。
    """
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _status_value(raw) -> str:
    """把状态归一成小写字符串。

    上游 ``DocStatus`` 是 ``str, Enum``，但 ``str(DocStatus.PROCESSED)``
    得到的是 ``"DocStatus.PROCESSED"`` 而不是 ``"processed"``。
    """
    value = getattr(raw, "value", raw)
    return str(value).lower()


async def classify_insert(rag, doc_id: str, track_id: str) -> InsertOutcome:
    """回 ``doc_status`` 核实这份文档到底进没进索引。

    ``ainsert`` 从不因为拒收而抛异常，它只把结果写进 ``doc_status`` 并返回
    track_id。上游有三种落法：

    - **正常处理** → ``doc_status[doc_id].status == "processed"``
    - **处理失败** → ``doc_status[doc_id].status == "failed"``，带 ``error_msg``
    - **被判重** → **不会**写进我们传的 doc_id，而是另起一条 ``dup-<hash>``
      记录（``status=failed``、``metadata.is_duplicate=true``），
      挂在同一个 track_id 下 → 用 ``get_docs_by_track_id`` 才找得到

    第三种最危险：调用方看到「没报错」就以为入库了，实际那份文档根本没进索引。
    """
    store = getattr(rag, "doc_status", None)
    if store is None:
        # 没有 doc_status 可问（例如测试里的假 rag）→ 无法核实，保持乐观
        return InsertOutcome("inserted")

    record = await store.get_by_id(doc_id)
    if record:
        status = _status_value(_field(record, "status"))
        if status == "processed":
            return InsertOutcome("inserted")
        reason = (
            _field(record, "error_msg")
            or _field(record, "content_summary")
            or f"LightRAG 处理状态为 {status}"
        )
        return InsertOutcome("failed", str(reason))

    # 我们传的 doc_id 不存在 → 多半是被判重了（上游会另起 dup- 记录）
    try:
        same_track = await store.get_docs_by_track_id(track_id)
    except Exception:  # pragma: no cover - 上游接口变更时的兜底
        same_track = {}
    for _dup_id, dup in (same_track or {}).items():
        metadata = _field(dup, "metadata") or {}
        if metadata.get("is_duplicate"):
            kind = metadata.get("duplicate_kind", "duplicate")
            return InsertOutcome("skipped", f"与已入库文档重复（{kind}）")

    return InsertOutcome("failed", "LightRAG 未生成该文档的处理记录")


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

    LightRAG 按规范名（basename 去掉解析器提示段）归并引用，同名文件会互相
    覆盖，所以必须先拦下来。返回的是**规范名**。
    """
    root = Path(root)
    seen: dict[str, set[str]] = {}
    for path in paths:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.as_posix()
        seen.setdefault(canonical_basename(path), set()).add(relative)
    return sorted(name for name, relatives in seen.items() if len(relatives) > 1)


async def ingest(
    rag, root: Path, settings: EngineSettings
) -> IngestReport:
    """把 ``root`` 下的所有受支持文件写入索引。

    - 同名文件（不同目录）→ 计入 ``failed``，**但不影响其他文件入库**；
    - 已入库的文件 → 计入 ``skipped``；
    - 被上游判重（内容相同、文件名不同）→ 计入 ``skipped``；
    - 单个文件抽取或写入失败 → 计入 ``failed``，不影响其他文件。
    """
    root = Path(root)
    files = discover_files(root)
    path_map = load_path_map(settings.working_dir)

    # LightRAG 按文件名（basename）判重，且这个判重索引跨批次、跨语料目录生效。
    # 同一 basename 指向不同文件时，后一个不会被索引 —— 所以必须在这里拦下来，
    # 不能让 ainsert 静默丢掉文档。
    conflicts = set(find_basename_conflicts(files, root))
    for path in files:
        key = canonical_basename(path)
        known = path_map.get(key)
        if known is None:
            continue
        # 比的是 resolve 之后的真实路径，不是字符串：同一个文件用相对路径
        # 和绝对路径写出来也不该被判成「两个同名文件」。
        try:
            same_file = Path(known).resolve() == path.resolve()
        except OSError:  # pragma: no cover - 极少见的路径解析失败
            same_file = False
        if not same_file:
            conflicts.add(key)

    conflict_reason = (
        "文件名与已入库文档冲突（LightRAG 按文件名判重，同名文件无法共存）: "
        + ", ".join(sorted(conflicts))
        + "。请给冲突文件改名，或清空索引目录（RAG_WORKING_DIR）后重新摄入。"
    )

    inserted: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for path in files:
        relative = path.relative_to(root).as_posix()
        key = canonical_basename(path)

        # 冲突只拦冲突的那几个文件 —— 早先的实现在这里整批中止，
        # 没冲突的文件既没入库也没出现在报告里，等于凭空消失。
        if key in conflicts:
            failed.append((relative, conflict_reason))
            continue

        if key in path_map:
            skipped.append(relative)
            continue

        doc_id = build_doc_id(relative)
        try:
            text = extract_text(path)
            track_id = await rag.ainsert(
                text,
                ids=[doc_id],
                file_paths=[str(path)],
            )
        except Exception as exc:
            failed.append((relative, f"{type(exc).__name__}: {exc}"))
            continue

        # ainsert 不抛异常不代表入库成功 —— 回 doc_status 核实。
        outcome = await classify_insert(rag, doc_id, str(track_id))
        if outcome.status == "inserted":
            # 存 resolve 后的绝对路径：后续的冲突判定要拿它和别的路径比较，
            # 存相对路径的话，换个工作目录就判不出来了。
            path_map[key] = str(path.resolve())
            inserted.append(relative)
        elif outcome.status == "skipped":
            # 被判重的文件并没有以自己的名字进索引，不写进 path_map ——
            # 写进去会让检索时的来源还原指向一个并不存在的引用。
            skipped.append(relative)
        else:
            failed.append((relative, outcome.reason))

    save_path_map(settings.working_dir, path_map)
    return IngestReport(
        inserted=tuple(inserted),
        skipped=tuple(skipped),
        failed=tuple(failed),
        discovered=len(files),
    )
