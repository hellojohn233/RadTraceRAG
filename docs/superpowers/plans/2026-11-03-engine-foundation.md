# RadTraceRAG 引擎底座 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 RadTraceRAG 里搭一层语料无关的 LightRAG 引擎，让 `ingest → query` 全链路在本机可跑通，产出带证据 ID 的回答。

**Architecture:** 新增 `src/radtracerag/engine/` 子包，把「配置 → LLM/Embedding 组装 → 索引 → 检索 → 回答 → 评估」拆成六个单一职责模块；现有 `ui/`、`contracts.py`、`integrations/` 本轮不动。检索走 LightRAG 的 `aquery_data()` 拿结构化结果，生成由本项目自己调 LLM 完成，引用格式 `[E1]` 完全可控并可校验。

**Tech Stack:** Python 3.11 · lightrag-hku 1.5.7 · openai SDK（OpenAI 兼容）· pypdf · numpy · python-dotenv · pytest + pytest-asyncio

**Spec:** `docs/superpowers/specs/2026-11-03-engine-foundation-design.md`

## 测试辅助代码约定（每个测试文件自包含）

计划里出现的这些 helper **写在对应的测试文件顶部**，不要放进 conftest（conftest 只放 `write_full_env` / `settings` / `settings_factory`）：

```python
# tests/test_ingest.py 顶部
def _minimal_pdf_bytes() -> bytes:
    """构造一个只含一页、含可抽取文本的最小 PDF。"""
    import io
    from pypdf import PdfWriter
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()
```
> 若 pypdf 写出的空页抽不出文本，改为在 `tests/conftest.py` 里放一个**固定的最小 PDF 的 base64 常量**
> （生成一次后内联进仓库），避免每次跑测试都动态构造。

```python
# tests/test_retrieve.py 顶部
class _FakeRag:
    def __init__(self): self.calls = []
    async def aquery_data(self, query, param):
        self.calls.append((query, param))
        return _PAYLOAD
```

```python
# tests/test_ingest.py 顶部（给 ingest 用）
class _FakeRag:
    def __init__(self): self.inserted = []
    async def ainsert(self, text, ids=None, file_paths=None, **kw):
        self.inserted.append({"text": text, "ids": ids, "file_paths": file_paths})
        return "mock-track-id"
```

```python
# tests/test_answer.py 顶部
def _fake_complete(reply: str):
    async def _impl(prompt, system_prompt=None, settings=None):
        return reply
    return _impl
```

```python
# tests/test_cli.py 顶部
def _fake_settings_cls(tmp_path, missing=()):
    """返回一个 EngineSettings 的替身类，from_env 时报出缺失的 key。"""
    from radtracerag.engine.settings import EngineSettings
    class _Fake(EngineSettings):
        @classmethod
        def from_env(cls, env_file=None):
            raise _MissingKeys(missing)
    return _Fake
```
> `_MissingKeys` 是一个携带 `missing` 列表并被 CLI 捕获转成友好提示的异常；
> CLI 必须 `except` 它并打印 `缺少配置: <KEY>`，而不是打印堆栈。

---

## Global Constraints

以下约束适用于**每一个**任务，不再逐条重复：

- Python `>=3.11,<3.14`，包管理只用 `uv`；不引入 `llama-index` / `langchain` / `chromadb` / `qdrant-client` 等额外框架
- 引擎层允许依赖：`lightrag-hku`、`openai`、`pypdf`、`numpy`、`python-dotenv`、`pytest`、`pytest-asyncio`
- **任何被 git 跟踪的文件里不得出现密钥**；`.env` 必须在 `.gitignore` 中，`.env.example` 只放占位符
- **不写 `score` / 相似度字段** —— LightRAG 不返回该值，宁缺勿造
- 引擎代码**不得出现任何领域硬编码**（医学术语、SLAKE 路径、题型名一律不写进 `engine/`）
- `RAG_WORKING_DIR` 一律相对**项目根**解析，不用进程 CWD
- LightRAG 日志级别一律设为 `WARNING`（`log_level=30`），否则 CLI 输出被淹没
- 提交信息用中文 conventional commits 格式（`feat:` / `fix:` / `docs:` / `test:` / `chore:`），正文遵循仓库的 `chinese-git-commit` 规范

### 精确取值（实现时直接用，不要改）

| 项 | 值 |
|---|---|
| 新增包路径 | `src/radtracerag/engine/` |
| 配置类 | `engine.settings.EngineSettings` |
| 环境变量 | `LLM_BASE_URL` `LLM_API_KEY` `LLM_MODEL` `LLM_MAX_ASYNC` `LLM_TIMEOUT` `EMBEDDING_BASE_URL` `EMBEDDING_API_KEY` `EMBEDDING_MODEL` `EMBEDDING_DIM` `EMBEDDING_MAX_TOKEN_SIZE` `RAG_WORKING_DIR` `RAG_TOP_K` `RAG_CHUNK_TOP_K` `RAG_ENABLE_RERANK` |
| 默认值 | `LLM_MAX_ASYNC=4` `LLM_TIMEOUT=240` `EMBEDDING_MAX_TOKEN_SIZE=8192` `RAG_WORKING_DIR=rag_storage` `RAG_TOP_K=40` `RAG_CHUNK_TOP_K=20` `RAG_ENABLE_RERANK=false` |
| CLI 子命令 | `doctor` `ingest` `query` `eval` |
| `query --mode` 取值 | `naive` `local` `global` `mix`（默认 `mix`） |
| 支持的语料后缀 | `.txt` `.md` `.pdf` |
| `doc_id` 算法 | `sha1(相对项目根的路径字符串).hexdigest()[:16]` |
| 路径映射文件 | `<working_dir>/path_map.json` |
| 许可证 | MIT，版权行 `Copyright (c) 2026 hellojohn233` |
| 证据编号 | `E1`、`E2`…… 从 1 开始，按检索返回顺序 |
| 引用标记正则 | `\[E(\d+)\]` |

---

## Review Focus

spec 暗示、但没有任何一个任务的测试覆盖的失败模式。每一行都对应下面某个任务里已写好的测试：

1. **`.env` 缺失或缺某个 key** —— `doctor` 必须逐项报出缺哪个变量，不能抛原始 stack trace（→ Task 1 / Task 8）
2. **服务商返回的 embedding 维度 ≠ `EMBEDDING_DIM`** —— 必须在 `doctor` 阶段就报出**实际维度**，不能拖到建索引时炸（→ Task 3 / Task 8）
3. **语料目录不存在、为空、或全部文件被跳过** —— `ingest` 必须友好退出并说明原因，不能抛栈（→ Task 6）
4. **`--mode` 传了非法值** —— CLI 必须报错并列出四个合法取值（→ Task 8）
5. **模型编造引用标记**（答案里出现 `[E9]` 而本次只有 E1–E3）—— 必须从正文**移除**并计入 `unsupported_markers`，不能原样输出（→ Task 2 / Task 7）
6. **两个不同目录下的同名文件** —— 必须报错拒绝入库，不能静默让 LightRAG 覆盖（→ Task 6）
7. **同一文件重复 ingest** —— 必须识别为「跳过」而非「失败」，进程退出码仍为 0（→ Task 6）

---

---

### Task 1: 工程地基（依赖、配置、许可证）

**Files:**
- Modify: `pyproject.toml`（加 `pypdf` 依赖；加 `[dependency-groups] dev`）
- Modify: `uv.lock`（由 `uv sync` 自动更新）
- Create: `.env.example`
- Create: `LICENSE`
- Create: `src/radtracerag/engine/__init__.py`
- Create: `src/radtracerag/engine/settings.py`
- Create: `tests/test_settings.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: 无
- Produces:
  ```python
  # engine/settings.py
  @dataclass(frozen=True)
  class EngineSettings:
      llm_base_url: str
      llm_api_key: str
      llm_model: str
      llm_max_async: int
      llm_timeout: int
      embedding_base_url: str
      embedding_api_key: str
      embedding_model: str
      embedding_dim: int
      embedding_max_token_size: int
      working_dir: Path          # 已解析为绝对路径，基准=项目根
      top_k: int
      chunk_top_k: int
      enable_rerank: bool

      @classmethod
      def from_env(cls, env_file: Path | None = None) -> "EngineSettings": ...
      def missing_keys(self) -> list[str]: ...   # 缺失或非法的变量名，LLM_* 排在 EMBEDDING_* 前
  ```

- [ ] **Step 1: 加依赖并同步**

```bash
cd /Users/johnwick/projects/RadTraceRAG
uv add pypdf
uv add --dev "pytest>=8" "pytest-asyncio>=0.24"
uv sync
```

Expected: `pyproject.toml` 出现 `pypdf` 依赖和 `[dependency-groups] dev`；`uv run pytest --version` 输出版本号。

- [ ] **Step 2: 写失败的配置测试**

```python
# tests/test_settings.py
from pathlib import Path
from conftest import write_full_env
from radtracerag.engine.settings import EngineSettings

def test_from_env_reads_all_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_BASE_URL=https://llm.example/v1\n"
        "LLM_API_KEY=sk-llm\n"
        "LLM_MODEL=chat-x\n"
        "EMBEDDING_BASE_URL=https://emb.example/v1\n"
        "EMBEDDING_API_KEY=sk-emb\n"
        "EMBEDDING_MODEL=emb-y\n"
        "EMBEDDING_DIM=1024\n"
        "RAG_WORKING_DIR=rag_storage\n",
        encoding="utf-8",
    )
    s = EngineSettings.from_env(env_file)
    assert s.llm_model == "chat-x"
    assert s.embedding_dim == 1024
    assert s.top_k == 40 and s.chunk_top_k == 20
    assert s.enable_rerank is False
    assert s.llm_max_async == 4 and s.llm_timeout == 240
    assert s.embedding_max_token_size == 8192

def test_working_dir_is_absolute_relative_to_project_root(tmp_path):
    s = EngineSettings.from_env(_write_full_env(tmp_path))
    assert s.working_dir.is_absolute()
    assert s.working_dir.name == "rag_storage"

def test_missing_keys_lists_absent_variables(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_BASE_URL=https://llm.example/v1\n", encoding="utf-8")
    s = EngineSettings.from_env(env_file)
    missing = s.missing_keys()
    assert "LLM_API_KEY" in missing
    assert "LLM_MODEL" in missing
    assert "EMBEDDING_API_KEY" in missing
    assert "LLM_BASE_URL" not in missing
```

- [ ] **Step 3: 跑测试确认失败**

Run: `uv run pytest tests/test_settings.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'radtracerag.engine'`

- [ ] **Step 4: 实现 `engine/settings.py`**

要点：
- `from_env(env_file=None)`：默认读项目根 `.env`；用 `dotenv.load_dotenv(env_file, override=True)` 载入
- `working_dir` 解析：`project_root / value`（`project_root` 用 `Path(__file__).resolve().parents[3]`，与现有 `config.py` 的 `Settings.from_environment` 同基准）
- 整数字段用 `int(os.environ.get(...))` 并对非法值抛 `ValueError`（错误信息里带上变量名）
- `missing_keys()`：检查 6 个必填（`LLM_BASE_URL` `LLM_API_KEY` `LLM_MODEL` `EMBEDDING_BASE_URL` `EMBEDDING_API_KEY` `EMBEDDING_MODEL`）+ `EMBEDDING_DIM`（必须 > 0），返回顺序为 LLM_* → EMBEDDING_*

- [ ] **Step 5: 跑测试确认通过**

Run: `uv run pytest tests/test_settings.py -v`
Expected: 3 passed

- [ ] **Step 6: 写 `tests/conftest.py` 的共享 fixture**

后续所有任务的测试都依赖这个文件里的两个 helper，**必须在这一步建好**：

```python
# tests/conftest.py
"""所有测试共享的构造器。无网络、无 API key。"""
from pathlib import Path
import pytest
from radtracerag.engine.settings import EngineSettings

FULL_ENV = """\
LLM_BASE_URL=https://llm.example/v1
LLM_API_KEY=sk-llm-placeholder
LLM_MODEL=chat-placeholder
EMBEDDING_BASE_URL=https://emb.example/v1
EMBEDDING_API_KEY=sk-emb-placeholder
EMBEDDING_MODEL=emb-placeholder
EMBEDDING_DIM=1024
"""


def write_full_env(tmp_path: Path, **overrides: str) -> Path:
    """写一份完整 .env，返回其路径。overrides 覆盖个别变量。"""
    body = FULL_ENV + "".join(f"{k}={v}\n" for k, v in overrides.items())
    env_file = tmp_path / ".env"
    env_file.write_text(body, encoding="utf-8")
    return env_file


@pytest.fixture
def settings(tmp_path: Path) -> EngineSettings:
    return EngineSettings.from_env(write_full_env(tmp_path))


@pytest.fixture
def settings_factory(tmp_path: Path):
    """返回一个函数：传 overrides 得到对应 EngineSettings。"""
    def _make(**overrides) -> EngineSettings:
        return EngineSettings.from_env(write_full_env(tmp_path, **overrides))
    return _make
```

并把 Task 1 Step 2 测试里的 `_write_full_env(tmp_path)` 改为 `write_full_env(tmp_path)`（从 conftest 导入）。

- [ ] **Step 7: 写 `.env.example` 与 `LICENSE`**

`.env.example` 内容严格按 Global Constraints 的变量表，`LLM_API_KEY=sk-your-key-here` / `EMBEDDING_API_KEY=sk-your-key-here` 占位，其余给默认值。
`LICENSE` 用标准 MIT 文本，版权行 `Copyright (c) 2026 hellojohn233`。

- [ ] **Step 8: 确认 `.env` 被忽略、`.env.example` 不被忽略**

Run: `cd /Users/johnwick/projects/RadTraceRAG && git check-ignore -v .env && git check-ignore -v .env.example; echo "exit=$?"`
Expected: 第一条打印 `.gitignore` 里匹配 `.env` 的规则；第二条**无输出**（`.env.example` 未被忽略）

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml uv.lock .env.example LICENSE src/radtracerag/engine/ tests/test_settings.py tests/conftest.py
git commit -m "feat(engine): 引入 pypdf 与测试依赖，新增配置层 EngineSettings"
```

---

### Task 2: 通用数据结构与引用校验

**Files:**
- Create: `src/radtracerag/engine/contracts.py`
- Create: `src/radtracerag/engine/markers.py`
- Create: `tests/test_markers.py`
- Create: `tests/test_contracts.py`

**Interfaces:**
- Consumes: 无
- Produces:
  ```python
  # engine/contracts.py
  @dataclass(frozen=True)
  class Document:
      path: Path
      doc_id: str
      media_type: str

  @dataclass(frozen=True)
  class Evidence:
      evidence_id: str      # "E1" / "E2" ...
      chunk_id: str
      reference_id: str
      content: str
      source: str           # 完整路径

  @dataclass(frozen=True)
  class RetrievalStep:
      stage: str
      detail: str
      elapsed_ms: int = 0
      count: int = 0

  @dataclass(frozen=True)
  class Answer:
      text: str
      evidence_ids: tuple[str, ...] = ()
      unsupported_markers: tuple[str, ...] = ()

  # engine/markers.py
  MARKER_PATTERN = re.compile(r"\[E(\d+)\]")

  def extract_markers(text: str) -> list[str]: ...
  def strip_markers(text: str) -> str: ...
  def validate_markers(text: str, valid_ids: Iterable[str]) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
      """返回 (移除全部标记后的正文, 有效标记元组, 无效标记元组)。"""
  ```

- [ ] **Step 1: 写失败的 marker 测试**

```python
# tests/test_markers.py
from radtracerag.engine.markers import extract_markers, strip_markers, validate_markers

def test_extract_markers_returns_unique_in_order():
    assert extract_markers("先看 [E2]，再看 [E1]，还是 [E2]。") == ["E2", "E1"]

def test_strip_markers_removes_all_markers():
    assert strip_markers("答案 [E1] 内容 [E2]") .strip() == "答案  内容"

def test_validate_markers_drops_unsupported_markers():
    text = "结论 [E1] 依据 [E9] 还有 [E2]。"
    clean, valid, unsupported = validate_markers(text, ["E1", "E2"])
    assert "[E9]" not in clean
    assert valid == ("E1", "E2")
    assert unsupported == ("E9",)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_markers.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'radtracerag.engine.markers'`

- [ ] **Step 3: 实现 `engine/markers.py`**

`extract_markers` 用 `MARKER_PATTERN.findall` 后加 `"E"` 前缀并按首次出现顺序去重；`strip_markers` 用 `MARKER_PATTERN.sub("", text)`；`validate_markers` 先取全部标记，再按 `valid_ids` 集合分流，把无效标记从正文里**连同其紧邻空格一起**删掉（避免留下空洞），返回 `(clean_text, valid_tuple, unsupported_tuple)`。有效标记在正文里**保留**（读者需要看到 `[E1]` 对应哪条证据）。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_markers.py -v`
Expected: 3 passed

- [ ] **Step 5: 写 contracts 测试**

```python
# tests/test_contracts.py
from pathlib import Path
from radtracerag.engine.contracts import Answer, Document, Evidence, RetrievalStep

def test_evidence_has_no_score_field():
    assert "score" not in Evidence.__dataclass_fields__

def test_answer_defaults():
    a = Answer(text="hi")
    assert a.evidence_ids == () and a.unsupported_markers == ()

def test_document_fields():
    d = Document(path=Path("a.md"), doc_id="x", media_type="text/markdown")
    assert d.doc_id == "x"
```

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/test_contracts.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add src/radtracerag/engine/contracts.py src/radtracerag/engine/markers.py tests/test_markers.py tests/test_contracts.py
git commit -m "feat(engine): 新增通用数据结构与引用标记校验"
```

---

### Task 3: LLM 与 Embedding 组装

**Files:**
- Create: `src/radtracerag/engine/llm.py`
- Create: `tests/test_llm.py`

**Interfaces:**
- Consumes: `EngineSettings`（Task 1）
- Produces:
  ```python
  # engine/llm.py
  def build_llm_func(settings: EngineSettings) -> Callable[..., Awaitable[str]]: ...
  def build_embedding_func(settings: EngineSettings) -> EmbeddingFunc: ...
  async def embed_once(settings: EngineSettings, text: str) -> tuple[np.ndarray, int]: ...
  ```

- [ ] **Step 1: 写失败的 embedding 测试**

```python
# tests/test_llm.py
import numpy as np
import pytest
from radtracerag.engine.llm import build_embedding_func, embed_once
from radtracerag.engine.settings import EngineSettings

@pytest.mark.asyncio
async def test_embedding_func_declares_max_token_size(settings):
    s = settings
    func = build_embedding_func(s)
    assert func.embedding_dim == 1024
    assert func.max_token_size == 8192
    # 内层协程签名必须含 max_token_size，LightRAG 靠它注入截断
    import inspect
    inner = func.func
    assert "max_token_size" in inspect.signature(inner).parameters

@pytest.mark.asyncio
async def test_embed_once_returns_actual_dimension(monkeypatch, settings):
    s = settings
    captured = {}

    class _Resp:
        data = [type("D", (), {"embedding": [0.1] * 768})()]

    class _Client:
        async def embeddings(self, **kw):
            captured.update(kw)
            return _Resp()

    monkeypatch.setattr("radtracerag.engine.llm._make_client", lambda s: _Client())
    vec, dim = await embed_once(s, "hi")
    assert dim == 768
    assert vec.shape == (768,)
    assert captured["encoding_format"] == "float"   # 关键：不能是 base64
    assert captured["model"] == s.embedding_model
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'radtracerag.engine.llm'`

- [ ] **Step 3: 实现 `engine/llm.py`**

要点：
- **不复用 `lightrag.llm.openai.openai_embed`** —— 它硬编码 `embedding_dim=1536` 且固定发 `encoding_format="base64"`
- `_make_client(settings) -> AsyncOpenAI`，用 `base_url=settings.embedding_base_url, api_key=settings.embedding_api_key`；测试通过 monkeypatch 这个函数替换
- 内层协程：
  ```python
  async def _embed(texts: list[str], max_token_size: int | None = None) -> np.ndarray:
      resp = await client.embeddings.create(
          model=settings.embedding_model,
          input=texts,
          encoding_format="float",
      )
      return np.asarray([d.embedding for d in resp.data], dtype=np.float32)
  ```
  参数名必须叫 `max_token_size`（LightRAG 的 `EmbeddingFunc.__call__` 按签名检查注入截断参数）
- `build_embedding_func` 返回 `EmbeddingFunc(embedding_dim=settings.embedding_dim, func=_embed, max_token_size=settings.embedding_max_token_size, model_name=settings.embedding_model)`
- `embed_once`：调一次并返回 `(np.asarray(第一个 embedding), 实际维度)`
- `build_llm_func`：返回协程，内部调 `lightrag.llm.openai.openai_complete_if_cache(model=..., prompt=..., system_prompt=..., history_messages=..., base_url=settings.llm_base_url, api_key=settings.llm_api_key, **kw)`
  - ⚠️ `from lightrag.llm.openai import openai_complete_if_cache` 必须放在 `load_dotenv()` **之后**，且本模块 import 时不得触发 `lightrag.llm.openai` 的模块级常量读取早于 dotenv

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_llm.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/radtracerag/engine/llm.py tests/test_llm.py
git commit -m "feat(engine): 组装 LLM 与 Embedding 函数，显式使用 float 编码格式"
```

---

### Task 4: LightRAG 实例工厂

**Files:**
- Create: `src/radtracerag/engine/rag.py`
- Create: `tests/test_rag.py`

**Interfaces:**
- Consumes: `EngineSettings`（Task 1）、`build_llm_func` / `build_embedding_func`（Task 3）
- Produces:
  ```python
  # engine/rag.py
  def build_rag(settings: EngineSettings) -> LightRAG: ...

  @asynccontextmanager
  async def rag_session(settings: EngineSettings) -> AsyncIterator[LightRAG]: ...
  ```

- [ ] **Step 1: 写失败的工厂测试**

```python
# tests/test_rag.py
from radtracerag.engine.rag import build_rag

def test_build_rag_passes_expected_arguments(tmp_path, settings_factory):
    s = settings_factory()
    rag = build_rag(s)
    assert rag.working_dir == str(tmp_path)
    assert rag.llm_model_name == s.llm_model
    assert rag.embedding_func.embedding_dim == s.embedding_dim
    assert rag.top_k == s.top_k
    assert rag.chunk_top_k == s.chunk_top_k
    assert rag.log_level == 30          # WARNING
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_rag.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'radtracerag.engine.rag'`

- [ ] **Step 3: 实现 `engine/rag.py`**

```python
def build_rag(settings: EngineSettings) -> LightRAG:
    return LightRAG(
        working_dir=str(settings.working_dir),
        llm_model_func=build_llm_func(settings),
        llm_model_name=settings.llm_model,
        embedding_func=build_embedding_func(settings),
        top_k=settings.top_k,
        chunk_top_k=settings.chunk_top_k,
        llm_model_max_async=settings.llm_max_async,
        default_llm_timeout=settings.llm_timeout,
        log_level=30,
    )
```
`rag_session` 是 `@asynccontextmanager`，进入时 `await rag.initialize_storages()`，退出时 `await rag.finalize_storages()`，用 try/finally 保证异常时也落盘。

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_rag.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/radtracerag/engine/rag.py tests/test_rag.py
git commit -m "feat(engine): 新增 LightRAG 实例工厂与存储生命周期上下文"
```

---

### Task 5: 检索层

**Files:**
- Create: `src/radtracerag/engine/retrieve.py`
- Create: `tests/test_retrieve.py`

**Interfaces:**
- Consumes: `EngineSettings`（Task 1）、`Evidence` / `RetrievalStep`（Task 2）、`LightRAG`（Task 4）
- Produces:
  ```python
  # engine/retrieve.py
  @dataclass(frozen=True)
  class RetrievalResult:
      question: str
      mode: str
      evidences: tuple[Evidence, ...]
      steps: tuple[RetrievalStep, ...]
      metadata: dict[str, Any]

  def evidence_from_payload(payload: dict[str, Any], path_map: dict[str, str]) -> tuple[Evidence, ...]: ...
  async def retrieve(rag, question: str, mode: str, settings: EngineSettings,
                     path_map: dict[str, str]) -> RetrievalResult: ...
  ```

- [ ] **Step 1: 写失败的 payload 解析测试**

```python
# tests/test_retrieve.py
from radtracerag.engine.retrieve import evidence_from_payload

_PAYLOAD = {
    "status": "success",
    "data": {
        "entities": [], "relationships": [],
        "chunks": [
            {"reference_id": "1", "content": "肝负责代谢。", "file_path": "02_liver_function.md", "chunk_id": "d1-chunk-000"},
            {"reference_id": "1", "content": "胆囊储存胆汁。", "file_path": "02_liver_function.md", "chunk_id": "d1-chunk-001"},
            {"reference_id": "2", "content": "CT 原理。", "file_path": "03_ct_basics.md", "chunk_id": "d2-chunk-000"},
        ],
        "references": [
            {"reference_id": "1", "file_path": "02_liver_function.md"},
            {"reference_id": "2", "file_path": "03_ct_basics.md"},
        ],
    },
    "metadata": {"query_mode": "mix", "keywords": {"high_level": [], "low_level": []}},
}

def test_evidence_ids_are_sequential():
    ev = evidence_from_payload(_PAYLOAD, {})
    assert [e.evidence_id for e in ev] == ["E1", "E2", "E3"]

def test_source_falls_back_to_path_map():
    ev = evidence_from_payload(_PAYLOAD, {"02_liver_function.md": "samples/02_liver_function.md"})
    assert ev[0].source == "samples/02_liver_function.md"

def test_source_falls_back_to_basename_when_not_in_map():
    ev = evidence_from_payload(_PAYLOAD, {})
    assert ev[2].source == "03_ct_basics.md"

def test_evidence_has_no_score():
    ev = evidence_from_payload(_PAYLOAD, {})
    assert "score" not in ev[0].__dataclass_fields__
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_retrieve.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'radtracerag.engine.retrieve'`

- [ ] **Step 3: 实现 `evidence_from_payload`**

- 读 `payload["data"]["chunks"]`，按顺序编号 `E1, E2, ...`
- `reference_id` 取 chunk 的 `reference_id`；`source` 先查 `path_map`（key = chunk 的 `file_path`），查不到则用 chunk 的 `file_path` 本身（basename）
- **没有 score 字段**
- payload 缺 `data` 或 `chunks` 为空 → 返回空元组，不抛异常

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_retrieve.py -v`
Expected: 4 passed

- [ ] **Step 5: 写 `retrieve()` 的测试（用假 rag）**

```python
class _FakeRag:
    def __init__(self): self.calls = []
    async def aquery_data(self, query, param):
        self.calls.append((query, param))
        return _PAYLOAD

@pytest.mark.asyncio
async def test_retrieve_uses_aquery_data_and_disables_rerank(settings):
    rag = _FakeRag()
    s = settings
    res = await retrieve(rag, "肝做什么", "mix", s, {})
    assert res.mode == "mix"
    assert len(res.evidences) == 3
    query, param = rag.calls[0]
    assert query == "肝做什么"
    assert param.mode == "mix"
    assert param.enable_rerank is False       # 显式关，避免只拿到 warning
    assert param.top_k == s.top_k
```

- [ ] **Step 6: 实现 `retrieve()`**

要点：
- `param = QueryParam(mode=mode, enable_rerank=settings.enable_rerank, top_k=settings.top_k, chunk_top_k=settings.chunk_top_k)`
  - **不要设 `only_need_context` / `include_references`** —— `aquery_data` 不读它们
- 用 `time.perf_counter()` 记录耗时，产出 `RetrievalStep(stage="检索", detail=f"{mode} 模式", elapsed_ms=..., count=len(evidences))`
- `metadata` 直接透传 payload 的 `metadata`
- `rag.aquery_data` 必须用 `await`

- [ ] **Step 7: 跑测试确认通过**

Run: `uv run pytest tests/test_retrieve.py -v`
Expected: 5 passed

- [ ] **Step 8: Commit**

```bash
git add src/radtracerag/engine/retrieve.py tests/test_retrieve.py
git commit -m "feat(engine): 基于 aquery_data 实现检索层与证据编号"
```

---

### Task 6: 摄入层

**Files:**
- Create: `src/radtracerag/engine/ingest.py`
- Create: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `EngineSettings`（Task 1）、`Document`（Task 2）、`LightRAG`（Task 4）
- Produces:
  ```python
  # engine/ingest.py
  SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}

  @dataclass(frozen=True)
  class IngestReport:
      inserted: tuple[str, ...]
      skipped: tuple[str, ...]
      failed: tuple[str, ...]      # (相对路径, 原因) 的元组序列

  def discover_files(root: Path) -> list[Path]: ...
  def build_doc_id(relative_path: str) -> str: ...    # sha1[:16]
  def extract_text(path: Path) -> str: ...             # txt/md 直读；pdf 用 pypdf
  def load_path_map(working_dir: Path) -> dict[str, str]: ...
  def save_path_map(working_dir: Path, mapping: dict[str, str]) -> None: ...
  def find_basename_conflicts(paths: list[Path], root: Path) -> list[str]: ...
  async def ingest(rag, root: Path, settings: EngineSettings) -> IngestReport: ...
  ```

- [ ] **Step 1: 写失败的文件发现与 doc_id 测试**

```python
# tests/test_ingest.py
from radtracerag.engine.ingest import (
    build_doc_id, discover_files, extract_text, find_basename_conflicts, load_path_map, save_path_map,
)

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
    (tmp_path / "d1").mkdir(); (tmp_path / "d2").mkdir()
    (tmp_path / "d1" / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "d2" / "a.md").write_text("y", encoding="utf-8")
    conflicts = find_basename_conflicts(discover_files(tmp_path), tmp_path)
    assert conflicts == ["a.md"]

def test_path_map_roundtrip(tmp_path):
    save_path_map(tmp_path, {"a.md": "samples/a.md"})
    assert load_path_map(tmp_path) == {"a.md": "samples/a.md"}
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'radtracerag.engine.ingest'`

- [ ] **Step 3: 实现发现 / doc_id / 路径映射 / 冲突检测**

- `discover_files`：`root.rglob("*")`，按后缀过滤，排序返回
- `build_doc_id`：`hashlib.sha1(relative_path.encode()).hexdigest()[:16]`
- `load_path_map` / `save_path_map`：读写 `<working_dir>/path_map.json`（不存在时返回 `{}`；目录不存在时先建）
- `find_basename_conflicts`：按 `path.name` 分组，组内相对路径不唯一则返回该 basename

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 4 passed

- [ ] **Step 5: 写文本抽取测试**

```python
def test_extract_text_reads_markdown(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# 标题\n正文", encoding="utf-8")
    assert "正文" in extract_text(p)

def test_extract_text_reads_pdf(tmp_path):
    # 依赖 pypdf；用真实最小 PDF 字节（helper 定义在 tests/test_ingest.py 顶部）
    pdf = _minimal_pdf_bytes()
    p = tmp_path / "a.pdf"
    p.write_bytes(pdf)
    assert isinstance(extract_text(p), str)
```

- [ ] **Step 6: 实现 `extract_text`**

- `.txt` / `.md`：按 UTF-8 读，失败时回退 `errors="replace"`
- `.pdf`：`pypdf.PdfReader(path)`，逐页 `extract_text()`，`"\n\n".join` 拼接；页数为 0 时返回 `""`
- 解析失败抛 `ValueError`，消息里含文件路径与原因（由 `ingest` 捕获转成 `failed` 条目）

- [ ] **Step 7: 跑测试确认通过**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 6 passed

- [ ] **Step 8: 写 `ingest()` 的测试**

```python
@pytest.mark.asyncio
async def test_ingest_inserts_and_records_path_map(tmp_path):
    root = tmp_path / "corpus"; root.mkdir()
    (root / "a.md").write_text("肝负责代谢。", encoding="utf-8")
    settings = settings_factory(RAG_WORKING_DIR=str(tmp_path / "storage"))
    rag = _FakeRag()
    report = await ingest(rag, root, settings)
    assert report.inserted == ("a.md",)
    assert report.failed == ()
    assert rag.inserted  # 确实调过 ainsert
    assert load_path_map(settings.working_dir) == {"a.md": str(root / "a.md")}

@pytest.mark.asyncio
async def test_ingest_rejects_basename_conflict(tmp_path, settings_factory):
    root = tmp_path / "corpus"; (root / "d1").mkdir(parents=True); (root / "d2").mkdir()
    (root / "d1" / "a.md").write_text("x", encoding="utf-8")
    (root / "d2" / "a.md").write_text("y", encoding="utf-8")
    report = await ingest(_FakeRag(), root, settings_factory(RAG_WORKING_DIR=str(tmp_path / "s")))
    assert report.inserted == ()
    assert report.failed and "a.md" in report.failed[0][0]

@pytest.mark.asyncio
async def test_ingest_empty_dir_is_not_a_failure(tmp_path, settings_factory):
    root = tmp_path / "corpus"; root.mkdir()
    report = await ingest(_FakeRag(), root, settings_factory(RAG_WORKING_DIR=str(tmp_path / "s")))
    assert report.inserted == () and report.failed == ()
```

- [ ] **Step 9: 实现 `ingest()`**

要点：
- 目录不存在 → 抛 `FileNotFoundError`（由 CLI 捕获成友好提示）
- **先跑冲突检测**，有冲突 → 全部不插入，把冲突项放进 `failed`，返回
- 每个文件：`text = extract_text(path)`；`doc_id = build_doc_id(相对路径)`；调用
  `await rag.ainsert(text, ids=[doc_id], file_paths=[str(path)])`
  - ⚠️ `file_paths` 传**完整路径**，LightRAG 会截成 basename → 我们自己存 path_map 还原
- 同一 `doc_id` 已在 path_map 里 → 计入 `skipped`，**不算失败**
- 单文件异常 → 计入 `failed`（`(相对路径, 原因)`），继续处理下一个
- 结束后 `save_path_map`
- `IngestReport.failed` 的元素类型是 `tuple[str, str]`

- [ ] **Step 10: 跑测试确认通过**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 9 passed

- [ ] **Step 11: Commit**

```bash
git add src/radtracerag/engine/ingest.py tests/test_ingest.py
git commit -m "feat(engine): 实现语料摄入、PDF 抽取与路径映射"
```

---

### Task 7: 回答层

**Files:**
- Create: `src/radtracerag/engine/answer.py`
- Create: `tests/test_answer.py`

**Interfaces:**
- Consumes: `EngineSettings`（Task 1）、`Evidence`（Task 2）、`validate_markers`（Task 2）、`build_llm_func`（Task 3）
- Produces:
  ```python
  # engine/answer.py
  ANSWER_SYSTEM_PROMPT: str

  async def answer_question(question: str, evidences: Sequence[Evidence],
                             settings: EngineSettings) -> Answer: ...
  ```

- [ ] **Step 1: 写失败的回答测试**

```python
# tests/test_answer.py
from pathlib import Path
from radtracerag.engine.answer import answer_question
from radtracerag.engine.contracts import Evidence

_EV = [
    Evidence(evidence_id="E1", chunk_id="c1", reference_id="1", content="肝负责代谢和解毒。", source="samples/liver.md"),
    Evidence(evidence_id="E2", chunk_id="c2", reference_id="2", content="胆囊储存胆汁。", source="samples/gall.md"),
]

@pytest.mark.asyncio
async def test_answer_keeps_valid_markers(monkeypatch, settings):
    monkeypatch.setattr("radtracerag.engine.answer._complete", _fake_complete("肝负责代谢与解毒 [E1]。"))
    a = await answer_question("肝做什么", _EV, settings)
    assert a.evidence_ids == ("E1",)
    assert a.unsupported_markers == ()
    assert "[E1]" in a.text

@pytest.mark.asyncio
async def test_answer_drops_unsupported_markers(monkeypatch, settings):
    monkeypatch.setattr("radtracerag.engine.answer._complete", _fake_complete("结论 [E1]，另有 [E9]。"))
    a = await answer_question("肝做什么", _EV, settings)
    assert a.unsupported_markers == ("E9",)
    assert "[E9]" not in a.text
    assert a.evidence_ids == ("E1",)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_answer.py -v`
Expected: FAIL —— `ModuleNotFoundError: No module named 'radtracerag.engine.answer'`

- [ ] **Step 3: 实现 `engine/answer.py`**

要点：
- `ANSWER_SYSTEM_PROMPT` 要求：只用给定证据作答、每条结论后面用 `[E1]` 标注依据编号、证据不足时明确说无法确认、**不要编造编号**
- `_complete(prompt, system_prompt, settings) -> str`：包一层 `build_llm_func(settings)` 调用；测试通过 monkeypatch 这个函数替换
- 提示词正文按顺序编号列出证据：`[E1] (来源: samples/liver.md)\n<content>`
- 拿到回答后：`clean, valid, unsupported = validate_markers(raw, [e.evidence_id for e in evidences])`
  - `valid` 里只保留**确实存在于 evidences** 的编号（`validate_markers` 已保证）
  - `unsupported` 里的是模型编造的编号
  - **`clean` 里保留有效标记**，只移除无效标记
- 返回 `Answer(text=clean, evidence_ids=valid, unsupported_markers=unsupported)`
- 证据为空时：不调 LLM，直接返回 `Answer(text="没有检索到可用证据，无法回答。", evidence_ids=(), unsupported_markers=())`

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_answer.py -v`
Expected: 2 passed

- [ ] **Step 5: 写空证据测试**

```python
@pytest.mark.asyncio
async def test_answer_with_no_evidence_does_not_call_llm(monkeypatch, settings):
    def _boom(*a, **k): raise AssertionError("不应调用 LLM")
    monkeypatch.setattr("radtracerag.engine.answer._complete", _boom)
    a = await answer_question("肝做什么", [], settings)
    assert a.evidence_ids == ()
    assert "证据" in a.text
```

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/test_answer.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add src/radtracerag/engine/answer.py tests/test_answer.py
git commit -m "feat(engine): 实现基于证据的回答生成与编造引用剔除"
```

---

### Task 8: CLI 四个子命令

**Files:**
- Modify: `src/radtracerag/cli.py`（重写）
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: 全部上游模块（Task 1–7）
- Produces:
  ```python
  # cli.py
  def main(argv: list[str] | None = None) -> int: ...
  def cmd_doctor(argv) -> int: ...
  def cmd_ingest(argv) -> int: ...
  def cmd_query(argv) -> int: ...
  def cmd_eval(argv) -> int: ...
  ```

- [ ] **Step 1: 写失败的 CLI 参数测试**

```python
# tests/test_cli.py
import pytest
from radtracerag.cli import main

def test_no_subcommand_exits_with_usage_error(capsys):
    assert main([]) == 2

def test_invalid_mode_lists_valid_modes(capsys):
    rc = main(["query", "肝做什么", "--mode", "bogus"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "naive" in err and "local" in err and "global" in err and "mix" in err

def test_query_missing_env_reports_missing_keys(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr("radtracerag.cli.EngineSettings", _fake_settings_cls(tmp_path, missing=["LLM_API_KEY"]))
    # _fake_settings_cls 在本测试文件内定义：返回一个 from_env 缺失指定 key 的类
    rc = main(["query", "肝做什么"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "LLM_API_KEY" in out
```

- [ ] **Step 2: 跑测试确认失败**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL —— 旧 `cli.py` 没有这些子命令

- [ ] **Step 3: 重写 `cli.py`**

要点：
- `argparse`，`subparsers(dest="command", required=True)`
- `doctor`：
  1. 打印 Python 版本、`lightrag-hku` 版本
  2. `EngineSettings.from_env()`；若 `missing_keys()` 非空 → 逐行打印 `缺少配置: <KEY>`，**返回 1**
  3. 真实调一次 `embed_once(settings, "dimension probe")` → 打印实际维度；与 `settings.embedding_dim` 不符 → 打印实际维度与期望维度，**返回 1**
  4. 真实调一次 LLM（`build_llm_func` 调一个 1-token 提示词）→ 打印首 token 与耗时
  5. 检查 `settings.working_dir` 是否存在；不存在则提示「索引目录尚未创建，先跑 ingest」
  6. 全部通过 → 返回 0
- `ingest <path>`：路径不存在 → 打印 `语料目录不存在: <path>` 并返回 1；调用 `ingest()`；打印 inserted/skipped/failed 计数与逐条明细；**有 failed 才返回 1**
- `query <question> [--mode] [--top-k]`：缺配置 → 返回 1；调用 `retrieve` + `answer_question`；按设计文档 §8 的格式打印答案 / 证据 / 检索轨迹 / 引用校验行；返回 0
- `eval`：打印「评测集未构建」并返回 0
- **所有子命令的错误路径都不允许抛未捕获异常** —— 用 try/except 包住，打印可读信息并返回非 0

- [ ] **Step 4: 跑测试确认通过**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 3 passed

- [ ] **Step 5: 写 `ingest` 空目录的退出码测试**

```python
def test_ingest_empty_dir_returns_zero(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "empty"; corpus.mkdir()
    monkeypatch.setattr("radtracerag.cli.EngineSettings", _fake_settings_cls(tmp_path))
    assert main(["ingest", str(corpus)]) == 0
```

- [ ] **Step 6: 跑测试确认通过**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add src/radtracerag/cli.py tests/test_cli.py
git commit -m "feat(cli): 重写为 doctor/ingest/query/eval 四个子命令"
```

---

### Task 9: 演示语料、端到端 mock 测试与 README

**Files:**
- Create: `samples/01_abdominal_anatomy.md`
- Create: `samples/02_liver_function.md`
- Create: `samples/03_ct_imaging_basics.md`
- Create: `samples/04_spleen_pancreas.md`
- Create: `tests/test_pipeline_mock.py`
- Modify: `README.md`（更新「当前阶段」与快速开始）

**Interfaces:**
- Consumes: 全部上游模块（Task 1–8）
- Produces: 无新接口

- [ ] **Step 1: 写 4 份演示语料**

每份 300–600 字，包含标题、若干小节、明确的知识点句。
`02_liver_function.md` 必须包含「肝脏负责代谢和解毒」「肝脏合成胆汁」「肝细胞受损会导致胆红素升高等具体事实，供后续验证答案与证据一致性用。

- [ ] **Step 2: 写端到端 mock 测试**

```python
# tests/test_pipeline_mock.py
"""无 API key 的端到端验证：假 LLM + 假 embedding 跑通 ingest → retrieve → answer。"""
import hashlib
import numpy as np
import pytest
from lightrag.utils import EmbeddingFunc

@pytest.mark.asyncio
async def test_pipeline_end_to_end_with_mock(tmp_path, monkeypatch):
    from radtracerag.engine import ingest as ingest_mod
    from radtracerag.engine import retrieve as retrieve_mod
    from radtracerag.engine import answer as answer_mod
    from radtracerag.engine.rag import build_rag

    # 假 embedding：必须是 async，且维度严格等于 embedding_dim
    async def fake_embed(texts, max_token_size=None):
        out = []
        for t in texts:
            raw = np.frombuffer(hashlib.sha256(t.encode()).digest(), dtype=np.uint8).astype(np.float32) / 255.0
            out.append(np.resize(raw, 64))
        return np.vstack(out)

    monkeypatch.setattr("radtracerag.engine.rag.build_embedding_func",
                        lambda s: EmbeddingFunc(embedding_dim=64, func=fake_embed))
    ...
    # 断言：ingest 成功、retrieve 返回证据、answer 返回带标记的答案
```

要点：
- 假 embedding **必须是 async**，维度**严格等于** `embedding_dim`（否则报 `Embedding dimension mismatch detected`）
- 假 LLM 按 `system_prompt` 内容分支：实体抽取返回 `{"entities": [], "relationships": []}`；关键词抽取返回 `{"high_level_keywords": [...], "low_level_keywords": [...]}`；其余返回 `"MOCK ANSWER [E1]"`
- **不要 mock 掉 `aquery_data`**，让真实的 LightRAG 走一遍 —— 这才验证得了 `file_path` 被截 basename、`reference_id` 是文件级这两个行为

- [ ] **Step 3: 跑测试确认通过**

Run: `uv run pytest tests/test_pipeline_mock.py -v`
Expected: 1 passed

- [ ] **Step 4: 更新 README**

改三处：
1. 「当前阶段」段落 → 说明引擎层已可跑通，UI 仍是演示数据
2. 「快速开始」→ 改成 `cp .env.example .env` → 填两组配置 → `uv sync` → `uv run radtracerag doctor` → `uv run radtracerag ingest samples/` → `uv run radtracerag query "..."`
3. 「参与项目」段落 → 说明 LICENSE 已定为 MIT

**不要**改 README 里医学影像路线图的部分 —— 那是业务层的事，本轮不动。

- [ ] **Step 5: 全量测试 + 密钥泄漏检查**

Run:
```bash
cd /Users/johnwick/projects/RadTraceRAG
uv run pytest -q
git grep -nE "sk-[A-Za-z0-9]{16,}" -- . ':!*.lock' ; echo "grep-exit=$?"
```
Expected: 全部测试通过；grep 无命中（`grep-exit=1`）

- [ ] **Step 6: Commit**

```bash
git add samples/ tests/test_pipeline_mock.py README.md
git commit -m "feat(engine): 增加演示语料与无 key 端到端测试"
```

---

### Task 10: 人工验收清单（需要真实 API key，由使用者执行）

**Files:**
- Create: `docs/ACCEPTANCE.md`

**Interfaces:**
- Consumes: 全部
- Produces: 人工验收步骤文档

- [ ] **Step 1: 写验收清单**

`docs/ACCEPTANCE.md` 必须逐条对应设计文档 §9 的 6 条验收标准，每条给出：命令、期望输出、常见失败原因与处理。
必须包含：
- `.env` 的两组配置分别填什么（LLM 一组、Embedding 一组，各自有 base_url / key / model；`EMBEDDING_DIM` 必须是该 embedding 模型的实际维度）
- `doctor` 报维度不符时怎么办
- `query` 对比 `naive` 与 `mix` 时看哪几个数字（耗时、`metadata.processing_info` 的实体/关系统计、证据条数）
- **明确写出**：证据条数在两种模式下可能相同，这是预期行为，不是 bug

- [ ] **Step 2: Commit**

```bash
git add docs/ACCEPTANCE.md
git commit -m "docs: 补充真实 key 环境下的人工验收清单"
```

---

## 完成定义

- [ ] `uv run pytest -q` 全绿
- [ ] `git grep -nE "sk-[A-Za-z0-9]{16,}"` 无命中
- [ ] `engine/` 目录下 `grep -riE "slake|肝|ct\b|liver"` 无命中（确认无领域硬编码）
- [ ] `uv run radtracerag doctor` 在填好 `.env` 后返回 0
- [ ] `uv run radtracerag ingest samples/` 返回 0 且重复执行不新增重复 chunk
- [ ] `uv run radtracerag query "..."` 输出答案 + 证据 + 检索轨迹，答案含 `[E1]` 标记
- [ ] `uv run radtracerag query "..." --mode naive` 与 `--mode mix` 有可观察差异（耗时 / 图谱统计）
