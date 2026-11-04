"""语料摄入：发现文件 → 抽取文本 → 写入 LightRAG 索引 → 维护路径映射。

三件 LightRAG 不替我们做的事
----------------------------
1. **PDF 解析**：传 ``ids`` 进 ``ainsert`` 一定走 RAW 路径
   （``pipeline.py`` 里明确写着「ids outranks docs_format by design」），
   LightRAG 内置的 pypdf 解析只在服务端上传路径生效。所以 PDF 自己抽。
2. **完整路径**：LightRAG 会把 ``file_path`` 截成 basename，引用里因此
   看不到目录。我们自己维护 ``path_map.json``（basename → 路径 + 内容哈希），
   检索时用它还原来源，判冲突时用它分辨「同一个文件」和「另一个同名文件」。
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

from radtracerag.engine.settings import PROJECT_ROOT, EngineSettings

#: 支持摄入的文本类后缀
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

#: 路径映射文件名（存放在工作目录下）
PATH_MAP_FILENAME = "path_map.json"

#: 「索引里躺着的是旧版本」时的统一措辞。
#:
#: 上游的 ``doc_status`` 过滤只看键在不在、不看状态 → doc_id 一旦进过
#: doc_status 就**永久拒收重插**。所以内容改过之后，重跑 ingest 是没有用的，
#: 只有两条出路：**改名**（换个 basename 重新入库）或**重建索引**。
#: 这句话必须说清楚，否则用户会一遍遍重跑，然后一遍遍困惑。
STALE_REASON = (
    "内容已变化，索引里仍是旧版本；"
    "如需更新请给文件改名，或清空索引目录后重新摄入"
)

try:  # pragma: no cover - 上游改名时走兜底分支
    from lightrag.utils_pipeline import normalize_document_file_path
except Exception:  # pragma: no cover
    normalize_document_file_path = None

try:  # pragma: no cover - 上游改名时走兜底分支
    from lightrag.utils import sanitize_text_for_encoding
    from lightrag.utils_pipeline import compute_text_content_hash
except Exception:  # pragma: no cover
    sanitize_text_for_encoding = None
    compute_text_content_hash = None


def content_hash_of(text: str) -> str:
    """上游给文档内容算的哈希 —— 与 ``doc_status`` 里的 ``content_hash`` 同口径。

    上游在 ``pipeline.py`` 里是这么算的：``ids`` 路径先把正文过一遍
    ``sanitize_text_for_encoding``（去控制字符、去首尾空白），再取
    ``compute_text_content_hash``。**少了 sanitize 那一步哈希就对不上**
    —— 实测空白文件的记录里存的是 ``md5("")``，而不是空白串的哈希。

    有了这个哈希就能回答一个关键问题：**索引里现在躺着的是哪一版内容？**
    拿它和当前文件比一比，就知道索引是不是过期了。
    """
    if compute_text_content_hash is None or sanitize_text_for_encoding is None:
        return ""
    try:
        return compute_text_content_hash(sanitize_text_for_encoding(text or ""))
    except Exception:  # pragma: no cover - 上游行为异常时的兜底
        return ""



def _storable_path(path: Path) -> str:
    """存进 path_map 的路径形式。

    **项目内的语料存相对项目根的路径**：项目目录搬家（换机器、换路径）后，
    存绝对路径会把每个文件都误判成「同名冲突」，用户唯一的出路是清空索引
    重建 —— 而索引本身是可以搬家的。``build_doc_id`` 的文档字符串里也写着
    「同一个语料目录在不同机器上生成的 doc_id 一致，索引也可以跨机器搬运」。

    **项目外的语料存绝对路径**：那种情况下路径变了就是真的变了，
    LightRAG 也确实会按文件名判重。
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _resolve_stored(stored: str) -> Path:
    """把 path_map 里存的路径还原成绝对路径（相对路径按项目根解析）。"""
    candidate = Path(stored)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


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
class MappedFile:
    """path_map 里的一条记录。

    除了路径，还记着**内容哈希**。光比路径不够：文件从 A 目录搬到 B 目录，
    路径变了但它还是同一个文件 —— 上游按 basename 判重，它依然进不去索引。
    有了内容哈希才能分清「同一个文件搬了家」（跳过）和「另一个同名文件」
    （冲突）。上游自己也是这么分的：``doc_status`` 里有 ``content_hash``。

    ``alias`` 标记这条记录**不是**一次真正的入库：文件被上游判重了
    （内容与索引里另一份文档相同），上游没有以这个名字建索引。
    不区分的话，报告会说「已入库」，而实际上索引里根本没有这份文件 ——
    原件被删掉时，引用的来源路径就指向了一个不存在的文件。

    ``stale`` 标记**索引里躺着的是旧版本**：文件在入库之后被改过。
    这个标记必须显式存下来 —— 靠哈希比较推断不出来（我们一旦把当前文件
    的哈希写回映射，下一次就看不出索引过期了）。
    """

    #: 项目根相对（项目内）或绝对（项目外）
    path: str
    #: **索引里那一版**的文件字节 sha256（内容变过时**不更新**）
    sha256: str = ""
    #: True = 这份文件本身没进索引（上游判重），它的内容在索引里的另一个文件下
    alias: bool = False
    #: True = 文件在入库后被改过，索引里仍是旧版本
    stale: bool = False



def _file_digest(path: Path) -> str:
    """文件内容的 sha256。读不出就返回空串（此时退化成只比路径）。"""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:  # pragma: no cover - 读不了的文件
        return ""


@dataclass(frozen=True)
class IngestReport:
    """一次摄入的结果。"""

    #: 真的进索引了（相对路径）
    inserted: tuple[str, ...] = ()
    #: 本次没有提交的（相对路径, 原因）—— 已入库 / 内容已变化 / 内容在别的文件名下
    skipped: tuple[tuple[str, str], ...] = ()
    #: 内容与索引里另一份文档相同、上游只收了一份的（相对路径, 原因）—— 不算失败
    duplicates: tuple[tuple[str, str], ...] = ()
    #: 空文档：上游处理了但一个分块都没产生（相对路径, 原因）
    empty: tuple[tuple[str, str], ...] = ()
    #: (相对路径, 失败原因) 序列
    failed: tuple[tuple[str, str], ...] = ()
    #: 语料目录里发现的受支持文件总数（不管最终入库几个）
    discovered: int = 0


@dataclass(frozen=True)
class InsertOutcome:
    """一次 ``ainsert`` 的真实结果。

    ``status`` 取值：

    - ``inserted`` —— 真的进索引了
    - ``skipped`` —— 本次没有提交（已入库，或内容在索引里的另一个文件下）
    - ``stale`` —— 索引里躺着的是旧版本（文件改过了）
    - ``empty`` —— 上游处理了，但一个分块都没产生（空白文件）
    - ``failed`` —— 上游处理失败，``reason`` 里是上游的原始错误信息

    ``alias`` 表示「这份文件本身没进索引」—— 它的内容在索引里**另一个**
    文件名下。区别在于：alias 的文件不能说成「已入库」。
    """

    status: str
    reason: str = ""
    #: 上游判重的类型（``filename`` / ``content_hash``）；不是判重时为空
    duplicate_kind: str = ""
    #: 这份文件本身没进索引（内容在索引里的另一个文件下）
    alias: bool = False



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


async def _classify_filename_duplicate(
    store, doc_id: str, original_doc_id: str, current_hash: str
):
    """上游说「这个文件名已经在索引里了」—— 要分清是**哪一种**。

    上游的 ``duplicate_kind="filename"`` 有几种截然不同的情况：

    1. **就是我们自己上一轮插的那份**（``original_doc_id == doc_id``）。
       这份文档在 doc_status 里有一行 —— 它的真实 ``status`` 才是判据：
       - ``processed`` + ``chunks_count == 0`` → 空文档占位；
       - ``processed`` + 内容哈希对不上 → 索引里是**旧版本**；
       - ``failed`` → 上一轮就失败了，**索引里没有它**。
    2. **索引里这个文件名被另一份内容占着**（``original_doc_id != doc_id``）。
       我们的内容**根本没进索引** —— 上游按文件名判重，直接把我们丢了。

    ⚠️ 绝不能在这里说「已在索引中」而没有证据：上一轮失败的文档会被
    上游记成同名重复，说成「已入库」就等于把一个**从没进过索引**的文件
    报告成入库了 —— 而且这个误报还会被写进 path_map，永久生效。
    """
    try:
        own = await store.get_by_id(doc_id)
    except Exception:  # pragma: no cover - 上游接口变更时的兜底
        own = None

    if own is not None:
        status = _status_value(_field(own, "status"))
        if status == "processed":
            chunks_count = _field(own, "chunks_count", None)
            if isinstance(chunks_count, int) and chunks_count == 0:
                # 上游用空内容占了这个文件名。它**不会**重新处理这个 doc_id
                # （上游的 doc_status 过滤只看键在不在，不看状态）。
                return InsertOutcome(
                    "empty",
                    "空文档：上游以空内容占了这个文件名，不会再重新处理；"
                    "请给文件改名，或清空索引目录后重新摄入",
                    duplicate_kind="filename",
                )
            stored_hash = str(_field(own, "content_hash", "") or "")
            if stored_hash and current_hash and stored_hash != current_hash:
                # 索引里躺着的是旧版本 —— 文件改过了
                return InsertOutcome(
                    "stale",
                    "内容已变化，索引里仍是旧版本；"
                    "如需更新请给文件改名，或清空索引目录后重新摄入",
                    duplicate_kind="filename",
                )
            return InsertOutcome("skipped", "已入库", duplicate_kind="filename")

        if status == "failed":
            reason = (
                _field(own, "error_msg")
                or _field(own, "content_summary")
                or "上游处理失败"
            )
            return InsertOutcome(
                "failed",
                f"{reason}（上游已把这个文档标记为失败，不会自动重试；"
                "请给文件改名，或清空索引目录后重新摄入）",
                duplicate_kind="filename",
            )

        # pending / 其它状态：上游还没处理完，不能当成已入库
        return InsertOutcome(
            "failed",
            f"上游状态为 {status}，文档尚未处理完成；"
            "请稍后重试，或清空索引目录后重新摄入",
            duplicate_kind="filename",
        )

    # 没有我们自己的记录 → 这份文档从没被我们插进去过
    if original_doc_id and original_doc_id != doc_id:
        return InsertOutcome(
            "failed",
            "文件名已被索引里的另一份文档占用（LightRAG 按文件名判重），"
            "本次未入库；请给文件改名，或清空索引目录后重新摄入",
            duplicate_kind="filename",
        )

    # 判不出原文档是谁 —— 绝不能猜成「已在索引中」
    return InsertOutcome(
        "failed",
        "上游报告文件名重复，但找不到对应的文档记录，无法确认是否已入库；"
        "请给文件改名，或清空索引目录后重新摄入",
        duplicate_kind="filename",
    )


async def classify_insert(
    rag, doc_id: str, track_id: str, current_hash: str = ""
) -> InsertOutcome:
    """回 ``doc_status`` 核实这份文档到底进没进索引。

    ⚠️ 判据只能是**本次 track_id 下的记录**。

    上一轮跑留下的 processed 行还留在 doc_status 里 —— 直接 ``get_by_id``
    拿到的是那一行，把「本次被拒收」误判成「已入库」。实测：删掉
    ``path_map.json`` 再跑 ingest，报告说「新增 4」，上游却建了四条
    ``dup-*`` 失败记录，实际零入库。

    上游三种落法：

    - **正常处理** → 本次 track 下有我们这个 doc_id，``status == "processed"``
    - **被判重** → 本次 track 下只有一条 ``dup-<hash>`` 记录
      （``metadata.is_duplicate=true``），我们这个 doc_id 不在里面
    - **处理失败** → 本次 track 下有我们这个 doc_id，``status == "failed"``

    ``current_hash`` 是当前文件内容的上游口径哈希，用来核对**索引里躺着
    的是哪一版**。
    """
    store = getattr(rag, "doc_status", None)
    if store is None:
        # 没有 doc_status 可问（例如测试里的假 rag）→ 无法核实，保持乐观
        return InsertOutcome("inserted")

    try:
        same_track = await store.get_docs_by_track_id(track_id) or {}
    except Exception:  # pragma: no cover - 上游接口变更时的兜底
        same_track = {}

    # 判重记录优先：它说明本次插入被上游拒了
    for _dup_id, dup in same_track.items():
        metadata = _field(dup, "metadata") or {}
        if not metadata.get("is_duplicate"):
            continue
        kind = str(metadata.get("duplicate_kind", "duplicate"))
        original = str(metadata.get("original_doc_id", "") or "")
        if kind == "filename_conflict":
            # 上游说「已有多份文档共用这个文件名，得由人来定哪个是原件」——
            # 这是索引坏了，不能当成正常的跳过。
            return InsertOutcome(
                "failed",
                f"索引里已有多份文档共用这个文件名，需要人工修复（{kind}）",
            )
        if kind == "filename":
            return await _classify_filename_duplicate(
                store, doc_id, original, current_hash
            )
        # content_hash：内容与索引里另一份文档完全相同 → 内容在，名字不在
        return InsertOutcome(
            "skipped",
            "内容已在索引中（以其他文件名入库）",
            duplicate_kind=kind,
            alias=True,
        )


    # 本次 track 下有我们这个 doc_id → 这就是本次插入的真实结果
    record = same_track.get(doc_id)
    if record is None:
        record = await store.get_by_id(doc_id)
        # 拿到的可能是上一轮留下的行（track_id 对不上），不能当本次的结果
        if record is not None and _field(record, "track_id") != track_id:
            record = None

    if record is None:
        return InsertOutcome("failed", "LightRAG 未生成本次插入的处理记录")

    status = _status_value(_field(record, "status"))
    if status == "processed":
        # ⚠️ 上游对**空白文件**也会写 ``processed``，但 ``chunks_count`` 是 0 ——
        # 一个分块都没有，等于什么都没进索引。报成「已入库」是假的。
        chunks_count = _field(record, "chunks_count", None)
        if isinstance(chunks_count, int) and chunks_count == 0:
            return InsertOutcome(
                "empty",
                "空文档：上游没有产生任何分块（文件可能是空白）；"
                "上游已用空内容占了这个文件名，不会再重新处理",
            )
        return InsertOutcome("inserted")
    reason = (
        _field(record, "error_msg")
        or _field(record, "content_summary")
        or f"LightRAG 处理状态为 {status}"
    )
    return InsertOutcome("failed", str(reason))


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


def load_path_map(working_dir: Path) -> dict[str, MappedFile]:
    """读取路径映射；文件不存在时返回空字典。

    文件损坏时**会告警**再返回空字典 —— 静默返回空字典会让「已入库」的判断
    彻底失效：所有文件都会被当成没入库重新提交，而上游会按文件名判重、
    悄悄跳过。我们必须让人看见这张表坏了。
    """
    path = Path(working_dir) / PATH_MAP_FILENAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"警告: 路径映射文件损坏，已按空映射处理: {path}"
            f"（{type(exc).__name__}: {exc}）"
        )
        print(
            "      已入库的文件会被重新提交，LightRAG 会按文件名判重并把它们跳过。"
        )
        return {}
    if not isinstance(data, dict):
        print(f"警告: 路径映射文件格式不对，已按空映射处理: {path}")
        return {}

    mapping: dict[str, MappedFile] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            mapping[str(key)] = MappedFile(
                path=str(value.get("path", "")),
                sha256=str(value.get("sha256", "")),
                alias=bool(value.get("alias", False)),
                stale=bool(value.get("stale", False)),
            )
        else:
            # 旧格式：只有路径
            mapping[str(key)] = MappedFile(path=str(value))
    return mapping


def save_path_map(
    working_dir: Path, mapping: dict[str, MappedFile | str]
) -> None:
    """写出路径映射（原子替换，避免写一半被读到）。"""
    directory = Path(working_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / PATH_MAP_FILENAME
    temp = target.with_suffix(".json.tmp")
    payload = {
        str(key): (
            {
                "path": value.path,
                "sha256": value.sha256,
                "alias": value.alias,
                "stale": value.stale,
            }
            if isinstance(value, MappedFile)
            else {"path": str(value), "sha256": "", "alias": False, "stale": False}
        )
        for key, value in mapping.items()
    }
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
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
    - 已入库的文件 → 计入 ``skipped``（带原因）；
    - 内容在原地改过的文件 → 计入 ``skipped``，原因里明说「索引里仍是旧版本」；
    - 内容与索引里另一份文档相同（上游按内容判重）→ 计入 ``duplicates``；
    - 空文档（上游一个分块都没产生）→ 计入 ``empty``；
    - 上游按文件名判重、但原文档不是我们的 → 计入 ``failed``（内容根本没进去）；
    - 单个文件抽取或写入失败 → 计入 ``failed``，不影响其他文件。

    ⚠️ 上游 LightRAG 1.5.7 的 ``doc_status`` 过滤**只看键在不在，不看状态**
    —— doc_id 一旦进过 doc_status 就**永久拒收重插**。所以：
    空文档、失败过的文档、内容改过的文档都**没法靠重跑解决**，
    只能改名或重建索引。报告里必须把这一点说清楚。
    """
    root = Path(root)
    files = discover_files(root)
    path_map = load_path_map(settings.working_dir)

    # LightRAG 按文件名（basename）判重，且这个判重索引跨批次、跨语料目录生效。
    # 同一 basename 指向不同文件时，后一个不会被索引 —— 所以必须在这里拦下来，
    # 不能让 ainsert 静默丢掉文档。
    conflicts = set(find_basename_conflicts(files, root))
    # 已入库但路径变了、而且内容也一样的文件 → 同一个文件搬了家，跳过。
    # 内容不一样的才是真冲突（LightRAG 只能留一个）。
    relocated: dict[str, str] = {}
    # 路径没变但内容变了 → 索引里留的还是旧版本，必须明确告诉用户
    changed: dict[str, str] = {}
    # 别名文件（内容在索引里另一个文件名下）的内容变了 → 它的 basename 在
    # 上游是空闲的（当初是按**内容**判重的），可以重新提交一次。
    alias_changed: set[str] = set()
    for path in files:
        key = canonical_basename(path)
        known = path_map.get(key)
        if known is None:
            continue
        known_path = known.path if isinstance(known, MappedFile) else str(known)
        known_hash = known.sha256 if isinstance(known, MappedFile) else ""
        known_alias = bool(getattr(known, "alias", False))
        # 比的是解析之后的真实路径，不是字符串：同一个文件用相对路径
        # 和绝对路径写出来也不该被判成「两个同名文件」；项目整体搬家后
        # 相对路径跟着搬，也不该变成假冲突。
        try:
            same_file = _resolve_stored(known_path) == path.resolve()
        except OSError:  # pragma: no cover - 极少见的路径解析失败
            same_file = False
        digest = _file_digest(path)
        if same_file:
            # 同一个文件 —— 但内容可能改过。索引里留的还是旧版本，
            # 说「已入库」是骗人的，必须让用户看见。
            if known_hash and digest and digest != known_hash:
                if known_alias:
                    alias_changed.add(key)
                else:
                    changed[key] = digest
            continue
        # 路径变了但内容一样 → 同一个文件搬了家，上游照样按 basename 判重，
        # 它进不去索引 —— 当作跳过，不算冲突。
        if known_hash and digest and digest == known_hash:
            relocated[key] = digest
            continue
        conflicts.add(key)

    conflict_reason = (
        "文件名与已入库文档冲突（LightRAG 按文件名判重，同名文件无法共存）: "
        + ", ".join(sorted(conflicts))
        + "。请给冲突文件改名，或清空索引目录（RAG_WORKING_DIR）后重新摄入。"
    )

    inserted: list[str] = []
    skipped: list[tuple[str, str]] = []
    duplicates: list[tuple[str, str]] = []
    empty: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for path in files:
        relative = path.relative_to(root).as_posix()
        key = canonical_basename(path)

        # 冲突只拦冲突的那几个文件 —— 早先的实现在这里整批中止，
        # 没冲突的文件既没入库也没出现在报告里，等于凭空消失。
        if key in conflicts:
            failed.append((relative, conflict_reason))
            continue

        if key in path_map and key not in alias_changed:
            known = path_map[key]
            known_alias = bool(getattr(known, "alias", False))
            known_stale = bool(getattr(known, "stale", False))
            known_hash = known.sha256 if isinstance(known, MappedFile) else ""

            if key in relocated:
                # 同一个文件搬了家：更新映射里的位置，不重新摄入
                path_map[key] = MappedFile(
                    path=_storable_path(path),
                    sha256=relocated[key],
                    alias=known_alias,
                    stale=False,
                )
                save_path_map(settings.working_dir, path_map)
                skipped.append((relative, "已入库（文件位置已更新）"))
                continue

            if key in changed:
                # 内容改过了，索引里留的还是旧版本 —— 必须说清楚，
                # 不能拿「已入库」糊弄过去。path_map 里的哈希**不更新**
                # （它记的是「索引里那一版」的哈希），另外打上 stale 标记，
                # 这样即使 path_map 丢了、我们把当前文件的哈希写了回去，
                # 下一次仍然看得出索引已经过期。
                path_map[key] = MappedFile(
                    path=_storable_path(path),
                    sha256=known_hash,
                    alias=known_alias,
                    stale=True,
                )
                save_path_map(settings.working_dir, path_map)
                skipped.append((relative, STALE_REASON))
                continue

            if known_stale:
                if known_hash:
                    # 内容又和索引里那一版一致了 —— 用户把内容改回去了，
                    # 索引其实是最新的。
                    path_map[key] = MappedFile(
                        path=_storable_path(path),
                        sha256=known_hash,
                        alias=known_alias,
                        stale=False,
                    )
                    save_path_map(settings.working_dir, path_map)
                    skipped.append((relative, "已入库"))
                    continue
                # 索引里那一版的哈希我们不知道（path_map 丢过）→ 保守地
                # 继续报「索引里是旧版本」，直到用户重建索引。
                path_map[key] = MappedFile(
                    path=_storable_path(path),
                    sha256=known_hash,
                    alias=known_alias,
                    stale=True,
                )
                save_path_map(settings.working_dir, path_map)
                skipped.append((relative, STALE_REASON))
                continue

            if known_alias:
                # 这份文件本身没进索引（上游按内容判重），它的内容在索引里
                # 另一个文件下 —— 说「已入库」是假的。
                skipped.append((relative, "内容已在索引中（以其他文件名入库）"))
                continue

            skipped.append((relative, "已入库"))
            continue

        # 别名文件的内容变了：删掉别名记录，重新提交一次。
        # 上游当初是按**内容**判重才拒的，那个 basename 在上游是空闲的，
        # 新内容完全可以以这个名字入库 —— 不需要重建整个索引。
        if key in alias_changed:
            path_map.pop(key, None)

        doc_id = build_doc_id(relative)
        try:
            text = extract_text(path)
            current_hash = content_hash_of(text)
            track_id = await rag.ainsert(
                text,
                ids=[doc_id],
                file_paths=[str(path)],
            )
        except Exception as exc:
            failed.append((relative, f"{type(exc).__name__}: {exc}"))
            continue

        # ainsert 不抛异常不代表入库成功 —— 回 doc_status 核实。
        outcome = await classify_insert(rag, doc_id, str(track_id), current_hash)
        if outcome.status == "inserted":
            path_map[key] = MappedFile(
                path=_storable_path(path), sha256=_file_digest(path)
            )
            inserted.append(relative)
        elif outcome.status == "skipped":
            # 上游拒了这次插入。分两种：
            #
            # - alias=False → 就是我们这份文档（上一轮插过），它确实在索引里；
            # - alias=True  → 内容在索引里**另一个文件名**下，这份文件本身
            #   没有以这个名字进索引 —— 不能说成「已入库」。
            #
            # 两种都把 path_map 写回去 —— 否则每次重跑都要白跑一遍上游，
            # 而 path_map 丢失正是「本次 track 判定」出错的场景。
            path_map[key] = MappedFile(
                path=_storable_path(path),
                sha256=_file_digest(path),
                alias=outcome.alias,
            )
            if outcome.alias:
                duplicates.append((relative, outcome.reason))
            else:
                skipped.append((relative, outcome.reason))
        elif outcome.status == "stale":
            # 索引里躺着的是旧版本。path_map 里「索引里那一版」的哈希我们
            # 不知道（path_map 丢过），留空 + 打 stale 标记 —— 宁可一直
            # 提醒，也不要谎报「已入库」。
            path_map[key] = MappedFile(
                path=_storable_path(path),
                sha256="",
                stale=True,
            )
            skipped.append((relative, outcome.reason))
        elif outcome.status == "empty":
            # 空文档：上游处理了，但一个分块都没产生。
            # **不写 path_map** —— 写进去就等于永久标记「已入库」，
            # 而上游已经用空内容占了这个文件名，它不会再重新处理。
            empty.append((relative, outcome.reason))
            continue
        else:
            failed.append((relative, outcome.reason))
            continue

        # 每插一个就落一次盘：中途被打断也不至于丢掉整张映射表。
        #
        # 代价是 O(n²) 次写入（n 个文件 = n 次全量重写）—— 演示级语料
        # 完全无所谓，语料规模上来要改成「定期批量落盘」。
        save_path_map(settings.working_dir, path_map)

    save_path_map(settings.working_dir, path_map)
    return IngestReport(
        inserted=tuple(inserted),
        skipped=tuple(skipped),
        duplicates=tuple(duplicates),
        empty=tuple(empty),
        failed=tuple(failed),
        discovered=len(files),
    )
