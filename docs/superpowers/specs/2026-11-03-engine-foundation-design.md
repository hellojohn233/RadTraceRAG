# RadTraceRAG 引擎底座设计（Engine Foundation）

- 日期：2026-11-03
- 状态：待批准
- 范围：通用 LightRAG 引擎 + CLI + 内置小语料，**本轮不接 UI，不碰 SLAKE，不做评估集**

---

## 1. 意图与为什么

RadTraceRAG 的业务方向已确定为**医学影像问答**（见 `doc/product-direction.md`）。但当前仓库只有 UI 外壳（1285 行里 1116 行是 `ui/`）和设计文档，**没有真实 pipeline**：无数据目录、无索引构建、无检索、无回答，界面跑的是 `ui/demo_data.py` 里的合成数据。

本轮目标：**先有一台能跑的发动机**。

具体说，用一个通用的 LightRAG 引擎把「语料 → 索引 → 检索 → 证据 → 回答」这条链路打通，让它对任意文本语料成立，同时不绑定任何特定数据集。医学影像作为业务应用是后续的事；本轮交付的引擎必须能独立验证。

**为什么值得先做引擎而不是先接业务**：SLAKE 的下载、影像解析、VLM 推理链路重（24GB 内存跑 7B VLM 很慢），而且上一轮实测发现 SLAKE 2094 道题里只有 267 道（12.7%）是知识题，纯视觉题不需要检索——评测协议要重新设计。先做通用引擎可以在没有这些负担的情况下把核心链路验证干净。

---

## 2. 目标 / 非目标 / 反目标

### 目标

1. `uv sync` 后填两个 key，就能用任意文本文件建索引并提问，出答案 + 证据 ID
2. 引擎层语料无关：txt / md / pdf 都能进；**引擎代码不含任何领域硬编码**
   （`samples/` 只是演示语料，不参与任何评测，也不算「内置数据集」）
3. LLM 与 Embedding **两组独立配置**，各自独立的 base_url / api_key / model
4. 四种检索模式（naive / local / global / mix）统一封装，可对比
5. 每条回答都能回溯到具体证据（chunk 级别），这是「可追溯」产品承诺的技术底座
6. CLI 四个子命令：`doctor` / `ingest` / `query` / `eval`

### 非目标（本轮明确不做）

- 不改 `ui/`（保持 `demo_data` 演示可用，等引擎稳定再接）
- 不下载 / 解析 SLAKE，不做影像链路，不调 VLM
- 不建评测集，不产出任何指标数字（`eval` 只是可运行的骨架）
- 不做多租户、不做鉴权、不做 FastAPI 服务
- 不引入 Qdrant/Milvus 等外部向量库（LightRAG 内置 nano-vectordb 够用，且不引入运行依赖）

### 反目标（做到就算失败）

- **引擎里出现任何 SLAKE / 医学领域的硬编码**（字段名、题型、路径都不得写死）
- **用 mock / 假数据假装跑通**——所有命令必须真的走 LLM 和向量库
- **密钥进仓库**：`.env` 必须被 gitignore，`.env.example` 只留占位符

---

## 3. 约束

| 约束 | 来源 | 影响 |
|---|---|---|
| Python 3.11–3.13 | `pyproject.toml` 已有约束 | 用 uv 管理，`.python-version` = 3.11 |
| `lightrag-hku>=1.5.7,<1.6` | 已锁定 1.5.7 | 必须按 1.5.7 的真实 API 写，不按网上旧教程 |
| **`openai` 不是 lightrag-hku 的依赖** | 实测：它靠 pipmaster 现场装，而 uv 的 venv 没有 pip → 必然失败 | 必须显式 `uv add openai`（已加，装到 3.42.1） |
| **单机、无 GPU、24GB 内存** | 实测 | 索引规模控制在几十到几百个 chunk；不引入本地大模型 |
| 现有 `contracts.py` 是医学专用 | 代码现状 | 通用数据结构另起 `engine/contracts.py`，不污染现有定义 |
| 现有 `ui/` 依赖 `radtracerag.contracts` | 代码现状 | 本轮不动 ui/，避免连带返工 |

### 已实测的 LightRAG 1.5.7 API（实现时以此为准）

```python
LightRAG(working_dir=..., embedding_func=..., llm_model_func=..., llm_model_name=...,
         top_k=40, chunk_top_k=20, ...)                    # 73 个参数

EmbeddingFunc(embedding_dim, func, max_token_size=None, send_dimensions=False,
               model_name=None, supports_asymmetric=False)

openai_complete_if_cache(model, prompt, system_prompt=None, history_messages=None,
                          base_url=None, api_key=None, ...) -> str
openai_embed(...) -> np.ndarray                           # 支持 EMBEDDING_BINDING_HOST / _API_KEY

await rag.ainsert(input, ids=None, file_paths=None) -> str
await rag.aquery(query, param=QueryParam(...)) -> str        # 只返回 LLM 回答正文
await rag.aquery_data(query, param=QueryParam(...)) -> dict  # ★ 返回结构化检索结果
await rag.initialize_storages()
await rag.finalize_storages()

QueryParam.mode ∈ {local, global, hybrid, naive, mix, bypass}
QueryParam.enable_rerank: bool = True      # ← 默认 True，无 rerank 模型会抛错
```

### `aquery_data` 的返回结构（1.5.7 实测）

```python
{
  "status": "success",
  "data": {
    "entities":      [{entity_name, entity_type, description, source_id,
                       file_path, created_at, reference_id}],
    "relationships": [{src_id, tgt_id, description, keywords, weight,
                       source_id, file_path, created_at, reference_id}],
    "chunks":        [{content, file_path, chunk_id, reference_id}],
    "references":    [{reference_id, file_path}],
  },
  "metadata": {
    "query_mode": "mix",
    "keywords": {"high_level": [...], "low_level": [...]},
    "processing_info": {total_entities_found, final_chunks_count, ...},
  },
}
```

### 五个必须踩准的坑（全部经源码核对 + mock 探针实测）

1. **rerank 缺配置不会抛异常，但结果没重排。**
   `utils.py:6187-6194` 是 `logger.warning(...)` 后直接 `return retrieved_docs`，**不抛错**。
   但默认 `enable_rerank=True` 而我们不提供 `rerank_model_func` → 只会拿到一条 warning 和未重排的结果。
   → 本轮**仍显式传 `enable_rerank=False`**（要的是确定性，不是报错）。

2. **`aquery` 会把检索上下文吞掉，`include_references` 在 SDK 路径下是死参数。**
   - `aquery()` 内部调 `aquery_llm()`，只返回 `llm_response["content"]`（`lightrag.py:3966-3970`）
   - `include_references` 全包**只在 `api/routers/query_routes.py`（HTTP API 层）被读取**，
     `lightrag.py` / `operate.py` / `naive_query` 从不读它
   - 即使 `only_need_context=True`，拿到的也只是渲染成字符串的 context 模板（引用块在**末尾**，
     格式 `[N] file_path`），正文是 JSON 块
   → **必须用 `rag.aquery_data()`**，它直接返回结构化数据，无需任何字符串解析。

3. **`reference_id` 是「文件级」的，不是「chunk 级」的。**
   LightRAG 按 `file_path` 去重后分配 ID（`utils.py:6996-7035`），
   同一文件的多个 chunk **共用同一个 `reference_id`**。
   → `Evidence` 必须同时带 `reference_id`（文件）+ `chunk_id`（片段）+ `content`。

4. **LightRAG 会把 `file_path` 归一化成 basename，丢掉目录。**
   实测：传入 `samples/02_liver_function.md`，`data.chunks[].file_path` 返回 `02_liver_function.md`。
   → 引用里**无法反推完整路径**，且不同目录下的同名文件会撞车。
   → **引擎自己维护 `doc_id → 完整路径` 映射**（落 `rag_storage/path_map.json`）；
     ingest 时检测 basename 冲突并报错。

5. **不能用 LightRAG 自带的 `openai_embed`。**
   - 它被 `@wrap_embedding_func_with_attrs(embedding_dim=1536)` 装饰，**维度硬编码 1536**，
     用 1024 维模型会抛 `Embedding dimension mismatch detected`
   - 它固定发 `encoding_format="base64"`（`openai.py:1189`）。该开关是**模块级常量**
     `EMBEDDING_USE_BASE64`（`openai.py:150`），**在 import 时读取**，
     而 LightRAG 核心**不会自己 `load_dotenv`** —— 多数国产 OpenAI 兼容端点不支持 base64，会直接 400
   → **自己写 embedding 函数**，用 `openai.AsyncOpenAI` 直连、显式 `encoding_format="float"`。

6. **传 `ids` 进 `ainsert` 一定走 RAW 路径，PDF 不会被解析。**
   `pipeline.py:1063-1065` 明确写着「`ids` outranks `docs_format` by design: explicit ids mark
   the sanitized body as RAW」。LightRAG 内置的 pypdf 解析只在服务端上传路径
   （`docs_format="pending_parse"`）生效，且 `pypdf` 不在依赖树内。
   → PDF 必须**我们自己抽文本**再 `ainsert(text, ids, file_paths)`。

---

## 4. 考虑过的做法

| 做法 | 取舍 | 结论 |
|---|---|---|
| **A. 通用引擎层 + 医学影像应用层** | 通用层可立刻用任意语料验证；医学层作 showcase。引擎跑通即有可展示的东西 | **采用** |
| B. 只做通用引擎，砍掉医学影像 | 最干净，但丢掉业务差异化，1285 行 UI 的影像逻辑作废，RadTraceRAG 品牌名也不再贴切 | 不采用 |
| C. 保持原样，只补医学影像 pipeline | 改动最小，但第一条路就被 SLAKE 和 VLM 推理锁死，与「先搭框架」的要求冲突 | 不采用 |
| D. 引入 Qdrant/Milvus | 符合企业偏好，但引入运行期外部依赖；LightRAG 内置 nano-vectordb 本轮足够 | 本轮不采用，留作后续扩展点 |

**为什么选 A**：唯一同时满足「能跑」和「不牺牲业务方向」的方案。引擎层是本轮交付物，业务层是下一轮的工作对象。

---

## 5. 架构

```text
        CLI (radtracerag)
              │
   ┌──────────┼───────────┐
   │          │           │
 doctor    ingest       query
   │          │           │
   │      ┌───┴────┐      │
   │      │ config │◄─────┘   读 .env，组装 LLM/Embedding 两组配置
   │      └───┬────┘
   │          │
   │      ┌───▼─────────────────────────┐
   │      │  engine/                    │
   │      │  ├── llm.py     组装 func    │
   │      │  ├── rag.py     LightRAG 工厂│
   │      │  ├── ingest.py  文件→索引    │
   │      │  ├── retrieve.py 检索+引用   │
   │      │  ├── answer.py  证据→回答    │
   │      │  ├── contracts.py 通用结构   │
   │      │  └── evaluate.py 评估骨架    │
   │      └──────────────┬──────────────┘
   │                     │
   │              LightRAG 1.5.7
   │        (nano-vectordb + networkx + json kv)
   ▼                     ▼
  .env            rag_storage/  (工作目录, gitignored)
```

### 目录

```text
src/radtracerag/
├── __init__.py
├── __main__.py
├── cli.py                    # 改：doctor / ingest / query / eval
├── config.py                 # 改：保留 Settings，新增 engine 配置读取
├── contracts.py              # 保留（医学层用，本轮不动）
├── engine/                   # ★ 新增：通用层
│   ├── __init__.py
│   ├── contracts.py          #   Document / Chunk / Evidence / RetrievalTrace / Answer
│   ├── llm.py                #   组装 llm_model_func + embedding_func
│   ├── rag.py                #   LightRAG 实例工厂 + 生命周期
│   ├── ingest.py             #   文件发现 / 切分 / 入库
│   ├── retrieve.py           #   四种模式 + 引用解析 + 轨迹
│   ├── answer.py             #   基于证据生成答案
│   └── evaluate.py           #   评估骨架（Recall@k / nDCG / EM 接口）
├── integrations/lightrag.py  # 保留
├── ui/                       # 本轮不动
└── evaluation/               # 保留

samples/                      # ★ 新增：内置小语料（进 git）
├── 01_abdominal_anatomy.md
├── 02_liver_function.md
├── 03_ct_imaging_basics.md
└── 04_spleen_pancreas.md

.env.example                  # ★ 新增
LICENSE                       # ★ 新增（MIT）
tests/                        # ★ 新增
docs/superpowers/specs/       # ★ 新增（本文档）
```

---

## 6. 配置设计

`.env.example`（**两组完全独立**）：

```dotenv
# ── LLM（chat / 实体抽取 / 关键词抽取 / 回答生成）
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=sk-xxxxxxxx
LLM_MODEL=your-chat-model
LLM_MAX_ASYNC=4
LLM_TIMEOUT=240

# ── Embedding（向量化）
EMBEDDING_BASE_URL=https://your-embedding-provider.example/v1
EMBEDDING_API_KEY=sk-xxxxxxxx
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_DIM=1024
EMBEDDING_MAX_TOKEN_SIZE=8192

# ── 运行参数
RAG_WORKING_DIR=rag_storage          # 一律相对**项目根**解析，不用 CWD
RAG_TOP_K=40
RAG_CHUNK_TOP_K=20
RAG_ENABLE_RERANK=false
```

**实现要点**

- 项目根 `.env`，用 `python-dotenv` 读取（已是 lightrag 传递依赖，无需新增）
- **不采用** LightRAG 自带的 `LLM_BINDING_*` / `EMBEDDING_BINDING_*` env 约定 —— 我们自己读、自己组装，配置名语义更清楚，且能明确区分两组 key
- `EMBEDDING_DIM` 必须与所选 embedding 模型的实际维度一致；`doctor` 命令会拿一次真实向量回来核对维度，不匹配立刻报错
- `.env` 已在 `.gitignore` 里（确认过）；`.env.example` 用占位符进 git

### 组装函数

```python
# engine/llm.py
import os
from dotenv import load_dotenv
load_dotenv()                       # 必须在 import lightrag 之前（见坑 5）
from lightrag.llm.openai import openai_complete_if_cache

async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kw) -> str:
    return await openai_complete_if_cache(
        model=settings.llm_model,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        **kw,
    )
```

**embedding 自己写，不复用 `openai_embed`**（坑 5）：

```python
# engine/llm.py
from openai import AsyncOpenAI

_client = AsyncOpenAI(base_url=settings.embedding_base_url, api_key=settings.embedding_api_key)

async def embedding_func(texts: list[str], max_token_size: int | None = None) -> np.ndarray:
    resp = await _client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
        encoding_format="float",        # 显式 float，多数兼容端点只支持这个
    )
    return np.asarray([d.embedding for d in resp.data], dtype=np.float32)

embedding_func = EmbeddingFunc(
    embedding_dim=settings.embedding_dim,
    func=embedding_func,
    max_token_size=settings.embedding_max_token_size,
    model_name=settings.embedding_model,
)
```

- **签名里带 `max_token_size` 参数是刻意的**：LightRAG 的 `EmbeddingFunc.__call__` 会做签名检查，
  命中就把 `max_token_size` 注入进来做自动截断（`utils.py` 的 `wrap_embedding_func_with_attrs` 机制）。
  不写这个参数 → 截断保护失效，超长文本会直接被服务端拒绝。
- `embedding_dim` 必须等于服务商模型的实际维度，`doctor` 会真实调一次并核对，不符立刻报错。

---

## 7. 各模块职责

### `engine/rag.py` — LightRAG 工厂

- `build_rag(settings) -> LightRAG`：构造实例，`working_dir` 指向 `RAG_WORKING_DIR`
- `async def session(settings)` 上下文管理器：进入时 `initialize_storages()`，退出时 `finalize_storages()`
- 存储用 LightRAG 默认：JsonKV / NanoVectorDB / NetworkX / JsonDocStatus —— **零外部服务**

### `engine/ingest.py`

- 递归发现文件，按扩展名白名单取 `.txt .md .pdf`
- PDF 由我们自己抽文本（`pypdf`）；LightRAG 内置解析在 `ainsert` 路径下不生效（坑 6）
- 调 `rag.ainsert(text, ids=[...], file_paths=[...])` —— **传 `file_paths` 让引用能回溯到源文件**
- **维护 `doc_id → 完整路径` 映射**（落 `rag_storage/path_map.json`），
  绕开 LightRAG 把 `file_path` 截成 basename 的行为（坑 4）
- **检测 basename 冲突**：两个不同目录下的同名文件 → 报错退出，不静默覆盖
- 输出：新增 / 跳过 / 失败的文件数与原因，**退出码非 0 表示有失败**

### `engine/retrieve.py`

- `retrieve(rag, question, mode) -> RetrievalResult`
- 调 **`rag.aquery_data(question, param)`**（不是 `aquery`），直接拿结构化结果，**不做任何字符串解析**
- `QueryParam(mode=mode, enable_rerank=settings.enable_rerank, top_k=..., chunk_top_k=...)`
- 从 `data.chunks` + `data.references` 构造 `Evidence[]`：
  - `evidence_id` = `f"E{index+1}"`（**我们自己的稳定编号**，与 LightRAG 的 `reference_id` 分开）
  - `source` = `reference_id → file_path` 映射得到
  - `content` = chunk 原文
  - **没有 score**：LightRAG 不返回 chunk 相似度，不编造
- 轨迹数据取自 `metadata`：查询模式、`keywords.high_level / low_level`、`processing_info.final_chunks_count`

### `engine/answer.py`

- **检索与生成本项目里是解耦的**：检索走 `aquery_data`，生成由我们直接调 LLM，**不经过 LightRAG 的 answer prompt**
- 输入：`question` + `Evidence[]`
- 提示词要求：**只用给定证据作答**，答案中用 `[E1]` `[E2]` 形式标注依据编号
- 后处理：解析答案里的每个 `[En]` 标记 → 校验 `En` 是否存在于本次返回的 `Evidence`；
  **不存在的标记一律丢弃并记入 `unsupported_markers`**（模型编造引用时能被捕获）
- 返回 `Answer(text, evidence_ids, unsupported_markers)` —— `unsupported_markers` 非空说明模型编了引用，这个字段本身是可观测性信号

### `engine/contracts.py` — 通用数据结构

```python
@dataclass(frozen=True)
class Document:      path: Path; doc_id: str; media_type: str

@dataclass(frozen=True)
class Evidence:      evidence_id: str      # 我们自己的 E1/E2/...
                     chunk_id: str        # LightRAG 的 chunk 唯一 ID
                     reference_id: str    # LightRAG 的文件级 ID
                     content: str
                     source: str          # 完整路径（从我们自己的映射表取，非 basename）
                     # ⚠️ 没有 score 字段：LightRAG 的任何公开返回路径都不给相似度。
                     #   宁可没有，也不编造分数。

@dataclass(frozen=True)
class RetrievalStep: stage: str; detail: str; elapsed_ms: int = 0; count: int = 0

@dataclass(frozen=True)
class Answer:        text: str
                     evidence_ids: tuple[str, ...] = ()
                     unsupported_markers: tuple[str, ...] = ()
```

### `engine/evaluate.py` — 骨架

- 接口先定死：`evaluate(retrievals, gold, k) -> dict`，返回 `recall@k / ndcg@k / mrr`
- 本轮只实现函数签名 + 一个用内置样本构造的**自检用**单元测试，**不产出任何对外指标数字**
- 评测集、标注规则、统计显著性检验留到后续轮次

---

## 8. CLI 契约

```bash
uv run radtracerag doctor
```
检查：Python 版本 · 依赖可导入 · `.env` 存在 · 两组配置齐全 · **真实调一次 embedding 并核对维度** · 真实调一次 LLM 并测延迟。任一失败退出码 1 并打印**具体缺什么**。

```bash
uv run radtracerag ingest samples/
```
建索引。幂等：同 `doc_id` 重复入库存为跳过。输出每个文件的处理结果。

```bash
uv run radtracerag query "肝脏的主要功能是什么？" --mode mix
```
`--mode ∈ {naive, local, global, mix}`（默认 `mix`）。输出：

```
答案
  <带 [E1][E2] 标记的回答>

证据
  [E1] samples/02_liver_function.md  (chunk_id=8f3a…, reference_id=1)
       <chunk 原文>

检索轨迹
  查询构造    12 ms
  向量检索    4 条 · 390 ms
  图扩展      4 条 · 540 ms
  引用解析    4 条 · 12 ms

引用校验  2 个标记全部有效
```

```bash
uv run radtracerag eval            # 本轮：报告「评测集未构建」，退出码 0
```

---

## 9. 验证步骤（完成标准）

```bash
cd /Users/johnwick/projects/RadTraceRAG
uv sync                                          # 含 openai（已加）
cp .env.example .env && $EDITOR .env             # 填两组配置
uv run radtracerag doctor                        # 必须全绿
uv run radtracerag ingest samples/               # 4 个文件入库
uv run radtracerag query "肝脏的主要功能是什么？" --mode mix
uv run radtracerag query "肝脏的主要功能是什么？" --mode naive   # 对比
uv run pytest -q                                 # 单测通过
```

**验收标准**

1. `doctor` 退出码 0，且**真的调用了** embedding 和 LLM（不是只检查配置存在）
2. `ingest samples/` 后 `rag_storage/` 下生成索引文件，重复执行不产生重复 chunk
3. `query` 输出答案 + ≥1 条证据 + 检索轨迹；答案里出现 `[E1]` 形式的标记
4. `naive` 与 `mix` 两种模式**有可观察差异**：耗时必然不同；此外 `mix` 的
   `metadata.processing_info` 会多出实体/关系统计而 `naive` 没有。
   （**不要求证据条数不同** —— 两者的增量在图谱实体上，chunk 层面可能几乎一致）
5. `pytest` 通过；`tests/` 至少覆盖：配置解析、引用解析、引用校验（模型编造标记时能被丢弃）
6. `git status` 中不含 `.env`，`grep -r "sk-" .` 在受版本控制的文件里无命中

---

## 10. 交付后仍待解决（不属于本轮）

- 医学影像应用层：`apps/medical/`，接 SLAKE + VLM
- 评测集设计与标注规则（**必须先解决「87% 题目不需要检索」的切分问题**）
- `mix` 模式在 SLAKE 小知识库（KG.zip 仅 0.1MB）上是否有收益 —— 需做规模探针
- README 与文档整体重写（当前仍描述医学影像全链路，与本轮后的实际形态有偏差）
- GitHub Social Preview 图（1280×640）
- `assets/export_png.sh` 里硬编码的 `/Users/viralt/Project/RadTraceRAG` 路径 —— **已修**（改为按脚本位置推导仓库根，
  并给 headless Chrome 加独立 `--user-data-dir`，避免参数被路由到用户已运行的 Chrome 实例）

---

## 11. 实现者还需要知道的事

### 无 key 也能验证整条链路的 mock 探针

本轮实测用一个假 LLM + 假 embedding 跑通了 `ainsert` → `aquery_data` 全流程（不需要任何 API key）。
实现时**把它固化成 `tests/test_pipeline_mock.py`**：

```python
# 假 embedding：必须是 async（EmbeddingFunc 会 await 它）
async def fake_embed(texts: list[str]) -> np.ndarray:
    out = []
    for t in texts:
        raw = np.frombuffer(hashlib.sha256(t.encode()).digest(), dtype=np.uint8).astype(np.float32) / 255.0
        out.append(np.resize(raw, DIM))     # sha256 只有 32 字节，必须 resize 到 DIM
    return np.vstack(out)

# 假 LLM：LightRAG 按 system_prompt 区分用途，返回对应形状的 JSON
async def fake_llm(prompt, system_prompt=None, history_messages=None, **kw):
    sp = (system_prompt or "").lower()
    if "entity" in sp or "gleaning" in sp:
        return json.dumps({"entities": [], "relationships": []}, ensure_ascii=False)
    if "keyword" in sp:
        return json.dumps({"high_level_keywords": ["liver"], "low_level_keywords": ["metabolism"]})
    return "MOCK ANSWER"
```

**四个实测踩出来的点**

1. `EmbeddingFunc.func` **必须是 async**，否则报 `TypeError: An asyncio.Future, a coroutine or an awaitable is required`
2. 假 embedding 的维度必须**正好等于** `embedding_dim`，否则报 `Embedding dimension mismatch detected`
3. `QueryParam` 必须显式 `enable_rerank=False`，否则抛 rerank 未配置异常
4. `data.chunks[].file_path` 返回的是 **basename**，完整路径要从我们自己的映射表取

这个探针同时是**单元测试的骨架**：解析逻辑、引用校验、路径映射都能在没有网络的情况下测。

### 其他

- **仓库不是 git 初始化的干净状态**：当前 `main` 分支有 2 个 commit，`git status` 干净。改完是否提交由人决定。
- **不要引入** `llama-index` / `langchain` / `chromadb` 等额外框架 —— 引擎层只用 LightRAG + openai SDK + pydantic/dataclasses + python-dotenv + numpy。
- **测试依赖要加**：`pyproject.toml` 里新增
  `[dependency-groups] dev = ["pytest>=8", "pytest-asyncio>=0.24"]`。
  当前仓库里 pytest / pytest-asyncio **都没装**，`uv run pytest -q` 现在跑不起来。
- **不要把密钥写进任何被 git 跟踪的文件**，包括测试夹具和文档示例。
- LightRAG 日志较多，`doctor` 和 `ingest` 需要把日志级别调到 `WARNING`（`log_level` 参数），否则 CLI 输出会被淹没。


---

## 12. 待决项（需要人拍板）

| # | 问题 | 选项 | 影响 |
|---|---|---|---|
| 1 | **PDF 支持吗？** | ✅ **已定：(a) 加 `pypdf`，我们自己抽文本** | ingest 多一段抽取逻辑 + 多一个依赖 |
| 2 | **验收第 4 条口径** | ✅ 已定：改为「耗时 + 图谱统计差异」，不要求证据条数不同 | 本轮定稿 |
| 3 | **embedding 服务商的模型名与维度** | 填进 `.env` 即可，`doctor` 会实测核对 | 不阻塞实现 |

> **设计文档状态：已于 2026-11-03 获批，开始实现。**

---

## 13. 审查记录（builder check，2026-11-03）

独立审查员核对了 lightrag-hku 1.5.7 的真实源码，结论：

**文档原有断言中被推翻的**

| 原断言 | 实际 |
|---|---|
| 无 rerank 模型会抛异常 | 只 warning，不抛错（`utils.py:6187-6194`） |
| `include_references=True` 时引用是输出第一行 | SDK 路径根本不读这个参数；引用在 context 末尾，且 `aquery` 会把 context 吞掉 |
| `openai_embed` 可直接用 | 它硬编码 1536 维、且固定发 `encoding_format=base64`，多数兼容端点会 400 |
| PDF 走 LightRAG 内置解析 | 传 `ids` 必走 RAW 路径，内置解析器不生效 |

**已被 mock 探针实测验证的（本轮新增证据）**

- `ainsert` → `aquery_data` 全链路跑通，返回结构与 §3 所写一致
- `reference_id` 确为文件级、`chunk_id` 形如 `doc-1-chunk-000`
- `file_path` 确被截成 basename
- `EmbeddingFunc.func` 必须是 async；假 embedding 维度必须严格等于 `embedding_dim`

**审查员提出、已在本轮定稿的（P2/P5/P6/P7/P8/P9）**

- `Evidence` **删除 `score` 字段** —— LightRAG 不返回相似度，宁缺勿造
- 验收第 4 条改为「耗时 + 图谱统计差异」，不要求证据条数不同
- ingest 用 `rag.doc_status` 出逐文件结果；duplicate 归为「跳过」不计失败；`doc_id` 用相对路径的稳定哈希
- 新增 `[dependency-groups] dev = [pytest, pytest-asyncio]`（当前仓库两者都没装）
- `RAG_WORKING_DIR` 一律相对项目根解析；`doctor` 检查索引目录存在且非空
- §2 目标 2 措辞改写：`samples/` 是演示语料，不算「内置数据集」
