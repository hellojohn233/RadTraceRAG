# 人工验收清单

引擎层需要在**真实 API key** 下验收。下面每一条都给出命令、期望输出和失败时的处理方式。

前置条件：`cp .env.example .env` 并填好两组配置。

---

## 0. 配置怎么填

`.env` 里是**两组互相独立**的配置，可指向不同服务商：

```dotenv
# LLM：负责实体抽取、关键词抽取、回答生成
LLM_BASE_URL=https://your-provider.example/v1
LLM_API_KEY=sk-...
LLM_MODEL=your-chat-model

# Embedding：负责向量化知识事实
EMBEDDING_BASE_URL=https://your-embedding-provider.example/v1
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL=your-embedding-model
EMBEDDING_DIM=1024
```

要点：

- `EMBEDDING_DIM` **必须等于该模型的实际输出维度**。bge-m3 是 1024，
  bge-large-zh-v1.5 是 1024，text-embedding-3-small 是 1536，
  text-embedding-3-large 是 3072。填错会在 `doctor` 阶段直接报出来。
- `LLM_*` 与 `EMBEDDING_*` 可以是同一个服务商的同一把 key，也可以是两把不同的 key。
- 绝大多数 OpenAI 兼容服务**不支持** `encoding_format=base64`，本项目已固定用
  `float`，不需要额外开关。
- 进程环境变量优先于 `.env` 文件，方便临时换模型：
  `LLM_MODEL=xxx uv run radtracerag query "..."`。

---

## 1. `doctor` —— 配置与服务连通性

```bash
uv run radtracerag doctor
```

**期望输出**（顺序固定）：

```
RadTraceRAG 0.1.0
Python 3.11.x
LightRAG 1.5.7
LLM      <模型名> @ <base_url>
Embedding <模型名> @ <base_url>
Embedding 维度: 实际 1024 / 配置 1024
LLM 响应: OK (312 ms)
索引目录: /.../rag_storage（尚未创建，先跑 ingest）
```

**失败处理**

| 现象 | 原因 | 处理 |
|---|---|---|
| `缺少配置: LLM_API_KEY` | `.env` 没填全 | 按提示逐项补齐 |
| `维度不一致: 服务商返回 768 维，EMBEDDING_DIM 配的是 1024` | `EMBEDDING_DIM` 填错 | 改成服务商文档里的实际维度 |
| `Embedding 调用失败: ...` | base_url / key 不对，或模型名不存在 | 先用 curl 单独打一次该服务的 embedding 接口 |
| `LLM 调用失败: ...` | base_url 少了 `/v1`、key 无效、或模型名错 | 同上 |

**验收标准 1**：`doctor` 退出码为 0，且**确实调用了** embedding 和 LLM（不是只检查配置存在）。

---

## 2. `ingest` —— 建索引

```bash
uv run radtracerag ingest samples/
```

**期望输出**：

```
新增 4 · 跳过 0 · 失败 0
  + 01_abdominal_anatomy.md
  + 02_liver_function.md
  + 03_ct_imaging_basics.md
  + 04_spleen_pancreas.md
```

再跑一次：

```bash
uv run radtracerag ingest samples/
```

**期望输出**：`新增 0 · 跳过 4 · 失败 0`，每行前缀 `=`，退出码 0。

**失败处理**

| 现象 | 原因 | 处理 |
|---|---|---|
| `文件名与已入库文档冲突 ...` | LightRAG 按文件名判重，同名文件无法共存 | 给其中一个改名，或清掉 `rag_storage/path_map.json` 后重来 |
| `! xxx.pdf: PDF 解析失败` | 文件损坏或加密 | 换文件，或先转成文字 |
| `语料目录不存在` | 路径写错 | 检查相对路径是相对**项目根**的 |

**验收标准 2**：`rag_storage/` 下生成索引文件；重复执行不产生重复 chunk、不返回非 0 退出码。

---

## 3. `query` —— 提问

```bash
uv run radtracerag query "肝脏的主要功能是什么？" --mode mix
```

**期望输出**分四段：

```
答案
  肝脏负责代谢与解毒，并合成胆汁 [E1]。

证据
  [E1] samples/02_liver_function.md  (chunk_id=..., reference_id=1)
       肝脏是维持血糖稳态的核心器官……

检索轨迹
  检索  mix 模式  1240 ms · 4 条
  命中  mix · 实体 7 · 关系 9  4 条

引用校验  1 个标记全部有效
```

**验收标准 3**：输出答案 + ≥1 条证据 + 检索轨迹；答案里出现 `[E1]` 形式的标记。

**失败处理**

| 现象 | 原因 | 处理 |
|---|---|---|
| `没有检索到可用证据，无法回答。` | 检索没命中，或还没 ingest | 先跑 ingest |
| 证据的 `source` 只有文件名没有目录 | `path_map.json` 丢失 | 重新 ingest |
| `已剔除 N 个不存在的编号` | 模型引用了不存在的证据 | **这是正常的观测信号**，说明模型在编造引用，引用编号已被从正文移除 |

---

## 4. `naive` 与 `mix` 的对比

```bash
uv run radtracerag query "肝脏的主要功能是什么？" --mode naive
uv run radtracerag query "肝脏的主要功能是什么？" --mode mix
```

**看这三个数字**：

1. **耗时** —— `mix` 必然更慢（多一轮关键词抽取 + 图扩展）
2. **`命中` 行的实体数与关系数** —— `naive` 只有 chunk，没有图谱统计；`mix` 会有实体/关系统计数
3. **`metadata.processing_info`** —— 两种模式的 `total_chunks_found` / `final_chunks_count`

> ⚠️ **证据条数在两种模式下可能相同，这是预期行为，不是 bug。**
> `mix` 相对 `naive` 的增量在**图谱实体与关系**上，不一定体现在 chunk 数量上。
> 本项目不保证图增强一定更准 —— SLAKE 的知识库很小（`KG.zip` 仅 0.1 MB），
> 图关系的长程优势可能发挥不出来。若实测下来 `mix` 没有优势，如实记录，这是
> 更有价值的结论。

---

## 5. 完整验收清单

逐条对照，全部满足才算通过：

- [ ] `uv run pytest -q` 全绿
- [ ] `git grep -nE "sk-[A-Za-z0-9]{16,}"` 无命中
- [ ] `grep -rniE "slake|liver|肝脏|medvqa" src/radtracerag/engine/ --include="*.py"` 无命中（引擎层无领域硬编码）
- [ ] `uv run radtracerag doctor` 退出码 0
- [ ] `uv run radtracerag ingest samples/` 退出码 0，重复执行退出码仍为 0 且全部计入「跳过」
- [ ] `uv run radtracerag query "..."` 输出答案 + 证据 + 检索轨迹，答案含 `[E1]` 标记
- [ ] `naive` 与 `mix` 在耗时或图谱统计上有可观察差异

---

## 6. 已知限制

| 限制 | 说明 |
|---|---|
| 图增强收益未验证 | SLAKE 知识库仅 0.1 MB，`mix` 是否优于 `naive` 需要实测 |
| 医学影像链路未接 | 引擎层不处理影像；VLM 适配器属于下一阶段 |
| 评测集未构建 | `radtracerag eval` 目前只报告「评测集尚未构建」 |
| 同名文件不能共存 | LightRAG 的限制，已在 ingest 阶段显式拦截 |
| 中文语料未覆盖 | 演示语料为中文，但无中文评测集 |
