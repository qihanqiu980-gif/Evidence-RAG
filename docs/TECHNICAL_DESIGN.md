# rag-app 技术设计文档

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档版本 | v0.8 |
| 文档状态 | M11 已完成 |
| 对应 PRD | `rag-app/docs/PRD.md` v0.8 |
| 技术阶段 | M11 知识发现 |

本文定义 `rag-app` 第一版的模块结构、数据模型、Provider 边界、问答编排、API/SSE 契约、CLI 和测试策略。实现如需偏离本文，应先更新文档并说明原因。

## 2. 设计原则

1. SQLite 和 uploads 是业务事实；Chroma 与 SQLite FTS5 关键词索引是可重建的派生索引。
2. 产品只有 online 运行模式；模型配置缺失时禁用入库与问答，不降级为本地规则模式。
3. 自动化测试通过 Fake Provider 与临时数据目录获得确定性，不依赖网络。
4. 知识证据问答内部固定执行拆分、检索、重排、语义判定、生成和校验。
5. Web API 只服务本地工作台；评测与运维能力通过 CLI 提供。
6. 模型输出非法、引用越权或校验失败时采用 fail-closed 局部拒答。
7. Chat、Embedding 和 Rerank 通过轻量 Provider Adapter 隔离，不引入 LangChain 或 LangGraph。
8. 文档格式差异被 Document Adapter 吸收，抽取后的纯文本走与 Markdown 相同的切块、指纹、向量化与检索流程。
9. 上传采用进程内线程池异步任务，不引入 Celery / Redis；任务状态可随进程重启丢失，启动时只清理遗留 spool，属显式取舍。
10. SQLite、uploads 与 Chroma 是同一个数据面，提供 backup / restore 快照作为迁移前的安全网。

## 3. 总体架构

```text
React + Vite Workbench
        │ HTTP / SSE
        ▼
FastAPI Application
        │
        ├── Knowledge Base / Document API
        ├── Async Upload Jobs
        ├── Knowledge Discovery Jobs
        ├── Evidence QA Workflow
        ├── Status API
        └── Static Frontend
        │
        ├── KnowledgeBaseManager
        │     ├── Document Adapter
        │     ├── SQLite Metadata
        │     ├── Uploads Store
        │     └── Chroma Vector Store
        │
        ├── UploadJobManager (ThreadPool + spool)
        ├── DiscoveryJobManager (ThreadPool + SQLite cache)
        │
        └── ModelProvider Adapter
              ├── OpenAI-compatible Chat
              ├── OpenAI-compatible Embeddings
              └── Rerank API
```

外层预留 workflow registry，但第一版只注册 `evidence_qa`：

```text
workflow registry
└── evidence_qa
    ├── decompose
    ├── retrieve
    ├── rerank
    ├── judge
    ├── generate
    ├── validate
    └── compose
```

未来可以新增文档总结、订单查询、确定性计算或轻量澄清 workflow，但不得改变 `evidence_qa` 的内部强制顺序。

## 4. 项目结构

```text
rag-app/
├── .env.example
├── .gitignore
├── README.md
├── pyproject.toml
├── assets/demo-docs/
├── data/
│   ├── app.db
│   ├── chroma/
│   └── uploads/
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
├── eval/cases.jsonl
├── src/rag_app/
│   ├── config.py
│   ├── errors.py
│   ├── logging.py
│   ├── models.py
│   ├── cli.py
│   ├── server.py
│   ├── api/
│   │   ├── app.py
│   │   ├── contracts.py
│   │   └── sse.py
│   ├── core/
│   │   ├── knowledge_base.py
│   │   ├── discovery.py
│   │   ├── adapters.py
│   │   ├── ingestion.py
│   │   ├── jobs.py
│   │   ├── retrieval.py
│   │   ├── workflow.py
│   │   └── registry.py
│   ├── providers/
│   │   ├── base.py
│   │   └── online.py
│   └── storage/
│       ├── sqlite.py
│       ├── uploads.py
│       ├── backup.py
│       └── chroma.py
└── tests/
    ├── conftest.py
    ├── fakes/provider.py
    ├── test_config.py
    ├── test_chunking.py
    ├── test_storage.py
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_workflow.py
    ├── test_api.py
    ├── test_cli.py
    ├── test_backup.py
    ├── test_adapters.py
    ├── test_jobs.py
    ├── test_pdf.py
    └── test_evaluation.py
```

约束：

- `data/` 被 Git 忽略，只保存当前 online 配置对应的业务数据。
- `assets/demo-docs/` 保存一键导入的 6 份固定 Markdown。
- Fake Provider 只放在 `tests/fakes/`，不进入产品代码。
- 前端构建产物输出到 `frontend/dist/`，由 FastAPI 托管。

## 5. 技术栈与配置

### 5.1 技术栈

- Python `>=3.12,<3.13`
- FastAPI、Uvicorn、Pydantic、ChromaDB、httpx、python-dotenv
- React、Vite、TypeScript
- 开发依赖：pytest、ruff、mypy

不引入 LangChain、LangGraph、Celery、Redis 或 Docker 强依赖。

### 5.2 环境变量

```dotenv
RAG_APP_API_KEY=
RAG_APP_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
RAG_APP_CHAT_MODEL=qwen3.7-plus
RAG_APP_EMBEDDING_MODEL=text-embedding-v4
RAG_APP_EMBEDDING_DIMENSION=1024

RAG_APP_RERANK_MODEL=qwen3-rerank
RAG_APP_RERANK_URL=https://maas.qianwenaiapi.com/compatible-api/v1/reranks

RAG_APP_DATA_DIR=data
RAG_APP_UPLOAD_MAX_BYTES=5242880
RAG_APP_UPLOAD_MAX_FILES=20
RAG_APP_UPLOAD_MAX_WORKERS=2
RAG_APP_CHUNK_SIZE=800
RAG_APP_CHUNK_OVERLAP=100
RAG_APP_RETRIEVAL_TOP_K=7
RAG_APP_CANDIDATE_MULTIPLIER=20
RAG_APP_HYBRID_RETRIEVAL=true
RAG_APP_RETRIEVAL_VECTOR_WEIGHT=1.0
RAG_APP_RETRIEVAL_KEYWORD_WEIGHT=1.0
RAG_APP_RETRIEVAL_RRF_K=60
RAG_APP_MAX_SUBQUESTIONS=5

RAG_APP_REQUEST_TIMEOUT_SECONDS=60
RAG_APP_MAX_RETRIES=1
RAG_APP_RETRY_BACKOFF_SECONDS=0.2
RAG_APP_LOG_LEVEL=INFO

# Optional evaluation cost estimation. Blank prices disable cost estimation.
RAG_APP_COST_CURRENCY=CNY
RAG_APP_EMBEDDING_INPUT_PRICE_PER_1K=
RAG_APP_CHAT_INPUT_PRICE_PER_1K=
RAG_APP_CHAT_OUTPUT_PRICE_PER_1K=
RAG_APP_RERANK_INPUT_PRICE_PER_1K=
RAG_APP_RERANK_OUTPUT_PRICE_PER_1K=
```

配置规则：

- API Key 可以为空，让服务启动并返回配置状态。
- API Key 为空时，知识库基础管理可用，上传、示例导入和问答禁用。
- 必须校验 `0 <= overlap < size`、`top_k > 0`、`multiplier > 0`、`1 <= max_subquestions <= 5`、`1 <= upload_max_workers <= 8`。
- 成本单价必须是非负有限数；货币使用 3 位字母代码。单价未配置、供应商未返回 usage，或 usage 无法拆分为输入 / 输出 token 时，评测成本输出 `null`，不得给出伪估算。
- 数据目录解析为绝对路径，默认位于项目根目录。

## 6. 领域模型与身份

```text
KnowledgeBase
├── id: UUID string
├── name: unique display name
├── created_at: UTC datetime
└── documents

Document
├── id: UUID string
├── kb_id
├── filename
├── content_hash: SHA-256 of normalized extracted text
├── storage_path: relative path
├── chunk_count
├── status: ready
└── created_at

Chunk
├── id: "{document_id}:{chunk_index:04d}"
├── document_id
├── kb_id
├── chunk_index
├── heading_path: list[str]
├── content
├── char_start
├── char_end
└── content_hash
```

规则：

- `kb_id` 和 `document_id` 使用 UUIDv4。
- 内容规范化步骤为：先按扩展名抽取文本（Markdown 保留标题结构），再统一换行、去除首尾空白、结尾补换行，最后计算完整 SHA-256。
- 同一知识库内 `(kb_id, content_hash)` 唯一。
- 不同知识库允许相同内容。
- 文件名仅用于展示，不参与内容身份。
- 失败文档不持久化为 ready 记录；上传响应返回本次失败项和错误码。
- 时间使用 UTC ISO-8601，前端按本地时区展示。

## 7. SQLite 设计

```sql
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE knowledge_bases (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    chunk_count INTEGER NOT NULL CHECK (chunk_count > 0),
    status TEXT NOT NULL DEFAULT 'ready' CHECK (status IN ('ready')),
    created_at TEXT NOT NULL,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    UNIQUE (kb_id, content_hash)
);

CREATE INDEX idx_documents_kb_id ON documents(kb_id);
CREATE INDEX idx_documents_content_hash ON documents(kb_id, content_hash);

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    kb_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index > 0),
    heading_path TEXT NOT NULL,
    content TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_kb_id ON chunks(kb_id);

CREATE TABLE vector_collections (
    id TEXT PRIMARY KEY,
    collection_name TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL CHECK (status IN ('building', 'active', 'retired')),
    embedding_model TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    created_at TEXT NOT NULL,
    activated_at TEXT,
    retired_at TEXT
);

CREATE TABLE discovery_runs (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    source_fingerprint TEXT NOT NULL,
    document_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    topic_count INTEGER NOT NULL,
    question_count INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
);

CREATE TABLE discovery_topics (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    kb_id TEXT NOT NULL,
    title TEXT NOT NULL,
    type TEXT NOT NULL,
    summary TEXT NOT NULL,
    document_ids TEXT NOT NULL,
    sort_order INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES discovery_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE
);

CREATE TABLE discovery_questions (
    id TEXT PRIMARY KEY,
    topic_id TEXT NOT NULL,
    question TEXT NOT NULL,
    source_chunk_ids TEXT NOT NULL,
    sort_order INTEGER NOT NULL,
    FOREIGN KEY (topic_id) REFERENCES discovery_topics(id) ON DELETE CASCADE
);

CREATE TABLE discovery_topic_coverage (
    topic_id TEXT PRIMARY KEY,
    chunk_count INTEGER NOT NULL,
    confidence REAL NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (topic_id) REFERENCES discovery_topics(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX one_active_vector_collection
ON vector_collections(status)
WHERE status = 'active';
```

职责：

- `knowledge_bases` 保存知识库业务身份和名称唯一性。
- `documents` 保存 ready 文档、同库去重、原文件路径和 chunk 统计。
- `chunks` 保存切块业务事实，用于审计、doctor 对账和重建。
- `vector_collections` 记录当前激活 Chroma collection、模型名和维度。
- discovery 表缓存每个 KB 最近一次主题地图；`document_ids` 与 `source_chunk_ids` 均为 JSON 数组。
- `heading_path` 保存 JSON 数组字符串。

SQLite 连接启用 `foreign_keys`、WAL 和 `busy_timeout`。业务 schema 当前为 v2；v1 备份恢复后自动迁移并创建 discovery 表。文档与 chunk 的写入使用显式事务。

## 8. Uploads 与 Chroma 设计

### 8.1 Uploads 路径

```text
data/uploads/{kb_id}/{document_id}{原扩展名}
```

规则：

- 路径只使用服务端 UUID 和规范化后的原扩展名，不使用用户文件名。
- API 不返回 `storage_path`。
- 删除知识库时删除对应 uploads 子目录。
- 所有文件路径 resolve 后必须仍位于 uploads 根目录内。

### 8.2 Chroma collection

- 使用 PersistentClient。
- 距离度量：cosine。
- Collection metadata 保存 `embedding_model`、`embedding_dimension`、`space` 和 `schema_version`。
- active collection 由 SQLite `vector_collections` 指向。
- 无 active collection 时创建空 active collection。
- 有 active collection 时必须校验模型名和维度。
- 模型或维度不一致时抛出 `VectorConfigurationMismatch` 并拒绝启动。
- Chroma collection 缺失时提示 doctor / rebuild。

### 8.3 Chroma chunk metadata

```text
id: chunk_id
document: chunk.content
embedding: provider embedding
metadata:
  kb_id
  document_id
  document_name
  heading_path: JSON string
  chunk_index
  content_hash
  schema_version
```

向量召回使用 `where kb_id $in selected_kb_ids`，候选数为 `top_k * candidate_multiplier`，默认 `7 * 20 = 140`。相似度展示值使用 `1 - cosine_distance`，重排分数来自 Provider。

删除文档按 `document_id` 删除向量，删除知识库按 `kb_id` 删除向量。Chroma 删除幂等。

### 8.4 关键词索引与混合检索

SQLite 内维护 `chunks_fts` 虚拟表：

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    tokenize='trigram'
);
```

设计要点：

- FTS 索引与 Chroma 一样是派生索引，不 bump 业务 schema 版本，旧备份恢复后可自动重建。
- `SQLiteStore` 初始化时幂等创建索引；索引行数与 chunks 不一致时自动全量重建。
- 文档写入、删除文档和删除知识库在同一个 SQLite 事务内同步维护 FTS 行。
- doctor 输出 `keyword_index` 计数；与 chunks 不一致时报告 `keyword_index_count_mismatch`。
- 查询侧使用 trigram BM25：ASCII / 数字 token 取长度 ≥3 的词；中文连续序列切为 3-gram；长度不足 3 的短词用 `instr` 子串兜底召回。

混合检索流程：

```text
vector candidates (Chroma, candidate_limit)
keyword candidates (FTS5 BM25, candidate_limit)
→ weighted RRF fusion
→ provider rerank
→ top_k evidence
```

RRF 公式为 `score(chunk) = Σ weight_source / (rrf_k + rank_source)`，默认 `rrf_k=60`，两路权重均为 1.0。`RAG_APP_HYBRID_RETRIEVAL=false` 时完全关闭关键词召回，保持 M9 及之前的纯向量行为。仅存在于关键词结果中的 chunk，向量相似度展示值记为 0.0。

## 9. Provider Adapter

### 9.1 Protocol

```python
class ModelProvider(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int,
    ) -> list[RerankItem]: ...

    def chat_json(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        task: str,
    ) -> dict[str, Any]: ...
```

说明：

- Provider 不了解 SQLite、Chroma 或 workflow 状态。
- Prompt 由 workflow 层定义。
- `rerank` 返回原始候选下标和分数，不直接操作业务对象。
- `task` 用于测试 Fake Provider 与日志分类，不作为安全边界。

### 9.2 Online Provider

`OnlineModelProvider` 使用 `httpx.Client`：

- Embedding：`POST {base_url}/embeddings`
- Chat：`POST {base_url}/chat/completions`
- Rerank：`POST {rerank_url}`
- 温度 0，默认超时 60 秒。
- 仅对连接错误或 5xx 重试；`RAG_APP_MAX_RETRIES` 取值 0-5。
- 重试采用指数退避：`RAG_APP_RETRY_BACKOFF_SECONDS * 2^attempt`，单次等待不超过 30 秒。
- Chat JSON 使用 `response_format={"type":"json_object"}`。
- Embedding 默认按 10 条批量。
- Provider 聚合请求次数、尝试次数、transport / 5xx 重试次数、最终失败分类和 usage token 数；指标不包含 API Key、请求头、prompt、完整响应体或本地路径。

外部异常统一转换为 `ProviderUnavailableError` 或 `ProviderResponseError`，安全消息不包含 API Key、请求头、完整响应体或本地路径。

### 9.3 Fake Provider

Fake Provider 只存在于测试：

- 使用词、中文单字和字符 bigram 生成确定性向量。
- 使用词面覆盖率重排。
- 按测试任务返回固定或参数化 JSON。
- 可注入连接失败、非法 JSON、越权引用和数字校验失败。
- 不读取 `.env`，不提供 CLI 开关，不写入产品 `data/`。

## 10. 文档摄取

### 10.1 流程

`DocumentAdapter` 按扩展名分派抽取器，支持 `.md`、`.txt`、`.pdf`、`.docx`、`.html`、`.htm`；抽取后的纯文本交给统一的 `PlainTextIngester`。

```text
validate filename and extension
→ extract text by extension
   （.md 保留标题结构，其余转为纯文本）
→ reject empty
→ normalize
→ SHA-256
→ same-KB duplicate check
→ heading-aware / plain chunking
→ recursive overflow splitting
→ embed chunks
→ save upload file (保留原扩展名)
→ add Chroma vectors
→ insert SQLite document and chunks
→ update active collection count
```

抽取器实现：

| 扩展名 | 实现 |
| --- | --- |
| `.md` | `MarkdownIngester`，标题感知切块 |
| `.txt` | 标准库 UTF-8 解码 |
| `.pdf` | `pypdf`，逐页 `extract_text` 后拼接 |
| `.docx` | 标准库 `zipfile` 读 `word/document.xml`，按段落拼接 |
| `.html` / `.htm` | 标准库 `HTMLParser`，跳过 `script` / `style` / `head` |

错误码：

| 检查 | 错误码 |
| --- | --- |
| 扩展名不在支持列表 | `invalid_extension` |
| 单文件超过 5 MB | `file_too_large` |
| 文本类文件 UTF-8 解码失败 | `invalid_utf8` |
| 抽取或规范化后为空 | `empty_document` |
| PDF 解析失败 | `pdf_parse_failed` |
| Word 解析失败 | `docx_parse_failed` |
| HTML 解析失败 | `html_parse_failed` |
| 解析器依赖不可用 | `unsupported_parser` |
| 同库内容重复 | `duplicate_document` |
| 切块结果为空 | `chunking_failed` |
| Embedding 数量或维度错误 | `embedding_failed` |
| 文件、向量或元数据写入失败 | `storage_write_failed` |

### 10.2 切块规则

1. 识别 ATX heading。
2. 维护标题层级并生成标题路径。
3. 标题前非空正文作为“正文”章节。
4. 空章节跳过。
5. 小章节直接成为 chunk。
6. 超长章节按段落、换行、中英文句读和标点寻找自然边界。
7. 找不到自然边界时按长度切分。
8. 相邻 chunk 保留 overlap。
9. 记录原文 `char_start` 和 `char_end`。

默认 `chunk_size=800`、`overlap=100`。

### 10.3 批量、异步与补偿

同步端点 `POST /api/knowledge-bases/{kb_id}/documents` 单次最多 20 个文件，逐个独立处理；一个文件失败不影响后续文件。全部成功返回 `201`，部分成功返回 `207`，全部失败按错误类型返回 `400` 或 `503`。

异步端点 `POST /api/knowledge-bases/{kb_id}/documents/async` 在创建 job 前统一校验数量、文件名、扩展名和单文件大小，全部通过后才把原始字节写入 `data/spool/` 并返回 `202` 与 `job_id`。`UploadJobManager` 使用有界 `ThreadPoolExecutor`（`RAG_APP_UPLOAD_MAX_WORKERS`，1–8）处理批次；任务结束时删除对应 spool 文件。`GET /api/jobs/{job_id}` 返回任务状态、逐文件状态、错误码与阶段进度。job 注册表只在内存中，服务重启后不再可查；`UploadJobManager` 初始化时删除 spool 根目录下的遗留普通文件和符号链接，遇到真实目录跳过、遇到单个删除失败不阻断启动。

取消采用协作式边界：

1. `POST /api/jobs/{job_id}/cancel` 对非终态 job 设置 `cancel_requested` 并返回当前快照。
2. 尚未开始运行的 job 直接把全部 item 标记为 `cancelled`。
3. 正在运行的 job 在下一个文件或下一个进度阶段边界停止；`完成` 边界不再抛出取消，避免把已持久化文档误标为取消。
4. 已完成 item 保留 `completed` 与 document 结果；未开始 item 标记为 `cancelled`。
5. job 终态为 `completed` 或 `cancelled`，spool 清理幂等。

服务端不保存重试语义。前端“重新提交失败文件”仅使用当前浏览器会话仍持有的 `File` 对象重新调用 async 提交端点，生成一个全新 job。

写入顺序：

```text
upload file
→ Chroma vectors
→ SQLite metadata
```

补偿时按 `storage_path` 定位并清理已写入的原始文件、向量和元数据；`UploadStore.delete_document` 只删除目标文件，父目录非空时忽略 `rmdir` 失败。

失败后反向清理：

```text
SQLite metadata
→ Chroma vectors
→ upload file
```

清理必须幂等。清理失败返回 `storage_inconsistent`，不得伪装成功。

### 10.4 知识发现

知识发现不参与证据问答编排，也不扩大回答边界；它只把已入库资料转换为可点击的问题入口。

```text
document ready
→ sample chunks (max 240)
→ discovery_topics：归纳 4-8 个主题
→ discovery_questions：每个主题生成 2-4 个问题
→ 服务端过滤 source_chunk_ids
→ fingerprint 事务校验
→ SQLite discovery cache
```

- `DiscoveryJobManager` 使用单线程进程内队列，不引入 Celery / Redis。
- 同步上传、异步上传完成、demo 导入和文档删除后自动调度或失效；知识库删除随外键级联清理。
- 主题规划输入文档名、heading path 和最长 220 字符摘录；问题生成输入主题下最多 8 个 chunk 和最长 700 字符摘录。
- 模型返回的 chunk ID 必须存在于输入集合；问题来源必须属于对应主题，否则丢弃该主题或问题。
- 数据面指纹由该 KB 全量 chunk 的 `id + content_hash` 计算。保存前在同一个事务内复核指纹，防止分析期间上传或删除造成旧地图覆盖新数据。
- `topic_coverage.chunk_count` 使用主题规划阶段绑定的真实 chunk 去重计数；每条问题仍保留自己的 `source_chunk_ids`。
- 打开页面只读 SQLite 缓存；前端对 pending / processing 每秒轮询，用户点击“重新分析”才强制提交新 job。
- 点击推荐问题仅把问题文本交给 `/api/chat/stream`，不携带预生成答案，也不绕过检索、判定、生成或校验。

## 11. 删除与重建

### 11.1 删除文档

```text
find document
→ delete Chroma by document_id
→ delete upload file
→ delete SQLite document and chunks
```

任一步失败即停止，保留可重试状态，并由 doctor 定位残留。

### 11.2 删除知识库

```text
find kb
→ delete Chroma by kb_id
→ delete uploads/{kb_id}
→ delete SQLite kb row
```

SQLite 外键级联删除 documents 和 chunks。

### 11.3 Rebuild

`rebuild` 只重建向量，不修改 uploads 和 SQLite chunk 定义。

```text
confirm --yes
→ read SQLite chunks
→ create building collection
→ batch embed
→ write vectors
→ verify count and metadata
→ mark new collection active
→ retire old collection
→ delete retired collection
```

要求：

- 新 collection 未验证前不改变 active 指针。
- 新 collection 失败时保留旧 active collection。
- 新 collection 激活后再清理旧 collection。
- 清理失败由 doctor 报告 retired 残留。
- 模型配置变更后的 rebuild 是用户显式切换向量空间的方式。

## 12. 证据问答 Workflow

### 12.1 阶段函数

```python
decompose(question, history) -> QuestionDecomposition
retrieve(subquestion, kb_ids) -> CandidateSet
rerank(query, candidates) -> RankedEvidenceSet
judge(subquestions, ranked_sets) -> EvidenceDecisions
generate(answerable_items) -> AnswerParts
validate(parts, decisions, evidence) -> ValidatedAnswerParts
compose(validated_parts, decisions) -> FinalAnswer
```

阶段函数不直接写 HTTP 响应。SSE 层只消费 workflow 输出的安全事件。

### 12.2 问题拆分

模型输出：

```json
{
  "standalone_question": "string",
  "subquestions": [{"text": "string"}]
}
```

程序规则：

- 空数组或非法 JSON 回退为单个原问题。
- 最多 5 项，超过截断并标记 `truncated=true`。
- 程序重新编号 `q1` 到 `q5`，忽略模型 ID。
- 子问题文本必须非空。

### 12.3 语义判定

模型输出：

```json
{
  "decisions": [
    {
      "subquestion_id": "q1",
      "answerable": true,
      "evidence_ids": [1],
      "missing_information": "",
      "reason": "string"
    }
  ]
}
```

程序校验：

- 每个子问题必须有判定。
- 引用编号必须属于该子问题候选集合。
- 越权引用、缺失判定或非法输出时，该子问题保守拒答。
- `supporting` 表示回答证据，`related` 表示相关但不足以直接回答。

### 12.4 生成与校验

生成模型只接收可回答子问题和各自允许证据：

```json
{
  "parts": [
    {
      "subquestion_id": "q1",
      "answer": "string with [1]",
      "citations": [1]
    }
  ]
}
```

确定性校验：

1. 至少一个引用。
2. 引用属于该子问题允许证据。
3. 去除引用标记后，回答中的阿拉伯数字必须能在证据原文中找到。

语义校验由模型逐项判断结论是否完全由允许证据支持。明确失败时携带原因重生成一次；第二次仍失败或输出非法，仅将失败子问题替换为固定局部拒答。不可回答子问题由程序直接局部拒答。

## 13. Web API 与 SSE

### 13.1 API

- `GET /api/status`
- `GET /api/knowledge-bases`
- `POST /api/knowledge-bases`
- `DELETE /api/knowledge-bases/{kb_id}`
- `GET /api/knowledge-bases/{kb_id}/documents`
- `POST /api/knowledge-bases/{kb_id}/documents`
- `POST /api/knowledge-bases/{kb_id}/documents/async` → `202 {job_id}`
- `DELETE /api/knowledge-bases/{kb_id}/documents/{document_id}`
- `GET /api/jobs/{job_id}` → 任务状态 + 逐文件结果/进度
- `POST /api/jobs/{job_id}/cancel` → `202` + 当前任务快照
- `GET /api/knowledge-bases/{kb_id}/discovery` → 主题地图 + 可问问题
- `POST /api/knowledge-bases/{kb_id}/discovery/analyze` → `202` + 当前 discovery 状态
- `POST /api/knowledge-bases/{kb_id}/import-demo`
- `POST /api/chat/stream`

通用错误：

```json
{
  "detail": {
    "code": "duplicate_document",
    "message": "当前知识库中已存在相同内容"
  }
}
```

常用状态码：

| 错误码 | HTTP |
| --- | --- |
| `validation_error` | 400 |
| `file_too_large` | 400 |
| `invalid_extension` | 400 |
| `invalid_utf8` | 400 |
| `empty_document` | 400 |
| `duplicate_document` | 409 |
| `not_found` | 404 |
| `upload_not_cancelable` | 409 |
| `configuration_missing` | 503 |
| `provider_unavailable` | 503 |
| `storage_inconsistent` | 500 |
| `pdf_parse_failed` | 400 |
| `docx_parse_failed` | 400 |
| `html_parse_failed` | 400 |
| `unsupported_parser` | 400 |
| `internal_error` | 500 |

### 13.2 Status DTO

```json
{
  "status": "ready",
  "knowledge_base_count": 0,
  "document_count": 0,
  "chunk_count": 0,
  "limits": {
    "max_files_per_upload": 20,
    "max_file_bytes": 5242880,
    "accepted_extensions": [".md", ".txt", ".pdf", ".docx", ".html", ".htm"]
  }
}
```

`status` 可为 `ready`、`configuration_missing` 或 `storage_inconsistent`。接口不返回模型名或 API Key。

### 13.3 DTO 名称与字段边界

| DTO | 字段 | 说明 |
| --- | --- | --- |
| `SystemStatus` | `status`, `knowledge_base_count`, `document_count`, `chunk_count`, `limits` | 全局状态与概览统计 |
| `KnowledgeBaseCreate` | `name` | 创建请求 |
| `KnowledgeBaseSummary` | `id`, `name`, `document_count`, `chunk_count`, `created_at` | 知识库列表与创建响应 |
| `DocumentSummary` | `id`, `kb_id`, `filename`, `status`, `chunk_count`, `created_at` | 文档列表 |
| `UploadedDocument` | `DocumentSummary` 字段加 `progress` | 批量上传成功项 |
| `DocumentUploadError` | `filename`, `code`, `message` | 批量上传失败或重复项 |
| `DocumentBatchResult` | `documents`, `errors` | 上传与示例导入响应 |
| `ChatRequest` | `question`, `history`, `kb_ids` | SSE 问答请求 |
| `UploadJobCreated` | `job_id` | 异步上传提交响应 |
| `UploadJobItemSummary` | `filename`, `status`, `code`, `message`, `progress`, `document` | 异步任务逐文件结果 |
| `UploadJobSummary` | `job_id`, `kb_id`, `status`, `created_at`, `updated_at`, `items` | 异步任务查询响应 |
| `DiscoveryQuestionSummary` | `id`, `question`, `source_chunk_ids` | 可问问题及真实来源 |
| `DiscoveryTopicSummary` | `id`, `title`, `type`, `summary`, `document_ids`, `chunk_count`, `confidence`, `updated_at`, `questions` | 主题地图节点 |
| `DiscoverySummary` | `kb_id`, `status`, `job_id`, `code`, `message`, `document_count`, `chunk_count`, `topic_count`, `question_count`, `analyzed_at`, `topics` | 知识发现状态与缓存结果 |
| `SseEvent` | `type`, `stage`, `message`, `sequence`, `payload` | SSE envelope |

约束：

- `DocumentSummary.status` 第一版只有 `ready`。
- `UploadJobSummary.status` 可为 `pending`、`processing`、`completed`、`cancelled`。
- `UploadJobItemSummary.status` 可为 `pending`、`processing`、`completed`、`failed`、`cancelled`；取消项使用 `upload_cancelled` 错误码，且 `document=null`。
- `DocumentBatchResult.documents` 只包含已成功入库的文档。
- 重复与失败只出现在 `errors` 中，不进入文档列表。
- `KnowledgeBaseSummary.chunk_count` 由该库 ready 文档累计得出。
- `DiscoverySummary.status` 可为 `not_analyzed`、`pending`、`processing`、`completed`、`failed`、`cancelled`；数据面指纹不匹配时返回 `not_analyzed` 且不返回主题。
- `DiscoveryQuestionSummary.source_chunk_ids` 非空且必须指向当前 KB 的真实 chunk。
- 删除成功返回 `204 No Content`，无响应体。

### 13.4 文档上传 DTO

```json
{
  "documents": [
    {
      "id": "uuid",
      "kb_id": "uuid",
      "filename": "manual.md",
      "status": "ready",
      "chunk_count": 18,
      "created_at": "2026-10-04T00:00:00Z",
      "progress": ["校验", "切块", "向量化", "保存", "完成"]
    }
  ],
  "errors": [
    {
      "filename": "copy.md",
      "code": "duplicate_document",
      "message": "当前知识库中已存在相同内容"
    }
  ]
}
```

### 13.5 问答请求

```json
{
  "question": "它支持 WiFi 6 吗？价格是多少？",
  "history": [
    {"role": "user", "content": "介绍一下这款路由器。"}
  ],
  "kb_ids": ["uuid"]
}
```

校验：问题 1-4000 字符；历史最多 100 条；`kb_ids` 去重后 1-50 个且必须存在。

### 13.6 SSE envelope

```json
{
  "type": "stage_started",
  "stage": "retrieve",
  "message": "开始检索子问题",
  "sequence": 1,
  "payload": {}
}
```

事件类型：

- `stage_started`
- `stage_completed`
- `evidence`
- `decision`
- `answer_final`
- `completed`
- `error`

阶段枚举：

```text
decompose, retrieve, rerank, judge,
generate, validate, regenerate, compose
```

阶段与 UI 展示文案的固定映射：

| `stage` | UI 文案 |
| --- | --- |
| `decompose` | 问题拆分 |
| `retrieve` | 检索 |
| `rerank` | 重排 |
| `judge` | 证据判定 |
| `generate` | 生成 |
| `validate` | 校验 |
| `regenerate` | 重新生成 |
| `compose` | 合成回答 |

`evidence` payload 中的每条证据包含 `reference_id`、`document_id`、`document_name`、`heading_path`、`chunk_index`、`content`、`similarity`、`rerank_score`、`subquestion_id`、`subquestion`、`support_status`。不包含本地路径、hash 或 prompt。

`decision` payload：

```json
{
  "decisions": [
    {
      "subquestion_id": "q1",
      "answerable": true,
      "evidence_ids": [1],
      "missing_information": "",
      "reason": "证据直接回答子问题"
    }
  ]
}
```

`answer_final` payload：

```json
{
  "answer": "完整答案",
  "refused": false,
  "truncated": false,
  "parts": [
    {
      "subquestion_id": "q1",
      "question": "子问题",
      "answer": "已验证答案 [1]",
      "citations": [1],
      "status": "verified",
      "refusal_reason": ""
    }
  ]
}
```

`answer_final` 只在全部生成与校验完成后发送，包含完整答案、分段答案、引用、拒答原因和 `truncated` 标记。未验证正文不得通过任何事件输出。`refused=true` 仅表示全部子问题拒答；存在至少一个 `verified` 部分时 `refused=false`，界面仍需展示局部拒答部分。

### 13.7 UI 状态映射

| API / SSE 状态 | UI 全局状态 |
| --- | --- |
| 本地请求未完成 | 连接中 |
| `SystemStatus.status=ready` | 已连接 |
| `SystemStatus.status=configuration_missing` | 配置错误 |
| `SystemStatus.status=storage_inconsistent` | 索引异常 |
| 网络失败、`internal_error` 或未分类异常 | 服务异常 |

UI 不发明新的后端状态值；`连接中` 只是前端本地加载态。

## 14. CLI

```bash
rag-app serve --host 127.0.0.1 --port 8010
rag-app doctor
rag-app doctor --json
rag-app rebuild --yes
rag-app eval --dataset eval/cases.jsonl --top-k 7
rag-app backup [目录]
rag-app restore <归档> --yes
```

### 14.1 serve

启动 FastAPI 并检查数据面。API Key 缺失时仍可服务 UI、status 和知识库基础管理；向量模型或维度不匹配时拒绝启动。

启动流程：

1. 初始化安全日志格式。
2. SQLite 已存在时先执行 doctor；错误阻断启动，警告允许启动。全新空目录跳过前置 doctor，按正常路径初始化。
3. 检查 `frontend/dist/index.html`；缺失时输出构建提示，但不影响 API 启动。
4. 构建 FastAPI container；数据面初始化失败返回安全错误码。
5. Uvicorn 使用应用侧日志配置。
6. 正常退出或启动失败时幂等关闭自持有的 Provider。

端口冲突返回本地启动失败提示，不输出底层异常堆栈或 API Key。

### 14.2 doctor

检查：

- SQLite schema 和统计。
- active vector collection。
- Chroma collection 存在与 metadata。
- SQLite chunk 与 Chroma vector 数量。
- 孤儿向量和缺失向量。
- upload 文件存在性。
- uploads 未登记文件。
- 删除后残留。

输出 `ok / warning / error` 和 JSON 明细。

### 14.3 backup / restore

`backup` 把 `app.db`（用 SQLite online-backup API 做一致性拷贝）、`uploads/`、`chroma/` 和 `manifest.json` 打包为单个 `.tar.gz`。默认输出到 `backups/`，文件名带 UTC 时间戳。

`restore` 校验归档里的 `manifest.json` schema_version 与当前一致后恢复数据目录；目标目录非空时必须显式 `--yes`。备份归档只包含数据面四类内容，**不包含 `.env`、prompt 或任何密钥**，这是安全底线。

### 14.4 rebuild

必须显式 `--yes`。只重建向量，保留业务事实。适用于索引损坏或用户确认切换 Embedding 模型。

### 14.5 eval

CLI eval 使用 online Provider，需要有效 API Key。Fake Provider 指标回归通过 pytest 内的评测用例执行，不提供产品级 fake 开关。

固定数据集为 `eval/cases.jsonl`，当前包含 14 条用例。每条包含：

| 字段 | 说明 |
| --- | --- |
| `id` | 稳定用例 ID |
| `question` | 用户问题 |
| `history` | 可选多轮历史 |
| `expected_document_prefixes` | 期望命中的文档名前缀列表 |
| `expected_refusal` | `none`、`partial` 或 `all` |

评测输出检索命中率、MRR、拒答期望通过率、总体通过率、证据范围泄漏数量、引用准确率、忠实度、case 与汇总延迟、token usage、可选估算成本，以及 Provider 重试 / 失败统计。延迟使用本地单调时钟；成本只基于供应商 usage 与 `.env` 中配置的每 1K token 单价。输出不包含完整答案、提示词、API Key 或本地路径。默认评测所有本地知识库，也可通过重复 `--kb-id` 指定范围。

M10 新增指标定义：

| 指标 | 定义 | 适用范围 |
| --- | --- | --- |
| `retrieval_mrr` | 最终证据列表中首条命中期望文档前缀的排名取倒数，未命中计 0，按 case 平均 | 配置了 `expected_document_prefixes` 的用例 |
| `citation_accuracy` | verified part 的引用指向期望文档前缀证据的比例；引用不存在或不属于期望文档计为不准确 | 全部含引用的回答 |
| `faithfulness_rate` | 独立 `faithfulness` judge（chat JSON）逐 part 判断回答是否完全由其引用证据支持；调用失败或输出非法按不支持计 | 全部 verified 且非空回答 |

忠实度 judge 与工作流内置 validate 相互独立：eval 使用单独的 system prompt 与 `task="faithfulness"`，其耗时与 token 计入该 case 的评测统计，但暂不计入 `passed` 判定，作为独立质量观测指标。引用准确率是文档前缀级别的严格口径：引用相关但非期望文档的证据会被计为不准确，用于检索对比而非人工评分替代。

## 15. 日志与安全

允许记录 stage、`kb_id`、`document_id`、chunk 数、provider 错误类型、耗时和 HTTP 状态。

禁止记录或返回：

- API Key 或 Authorization header。
- prompt。
- 完整模型响应。
- 上传全文。
- 本地绝对路径。
- 底层异常堆栈。

未预期异常统一转换为 `internal_error`，详细异常只进入服务端安全日志。

## 16. 测试策略

测试全部使用 `tmp_path`、Fake Provider 和显式 `build_app(settings, provider=fake)` 装配，不读取 `.env`，不访问网络。

| 层 | 覆盖 |
| --- | --- |
| Unit | 配置、hash、切块、SQLite、Chroma metadata、rerank、多格式抽取、spool 启动清理、job 取消边界、知识发现来源过滤与缓存失效 |
| Integration | 上传（同步 + 异步）、提交预检、取消 API、删除、doctor、rebuild、多库过滤、失败补偿、backup / restore、知识发现 API 与异步任务 |
| Workflow | 拆分、判定、生成、校验、一次重生成、局部拒答 |
| API contract | DTO、状态码、错误码、SSE schema、安全输出 |
| Frontend E2E | 知识地图渲染、来源绑定展示、点击可问问题进入证据问答 |
| Evaluation | Fake Provider 指标与固定 badcase |
| Online smoke | 真实配置连通性与人工验收 |

评测观测必须覆盖：

- transport 与 HTTP 5xx 重试计数。
- 最终失败请求与失败分类计数。
- Chat / Embedding / Rerank usage token 聚合。
- 配置单价后的成本估算。
- usage 缺失或价格缺失时成本必须为 `null`。
- 延迟分位数与 summary 聚合。

必须覆盖：

- 空、非 `.md`、非 UTF-8、超限文件拒绝。
- 异步提交前校验空文件名、错误扩展名、超限文件和超量文件，且不创建 job。
- 排队取消、阶段边界取消、已完成 item 保留和 spool 清理。
- 换行不同但内容相同的重复拦截。
- 跨库相同内容允许。
- 标题路径、超长切块和 overlap。
- Embedding 数量与维度校验。
- 文件、向量、元数据写入失败后的反向清理。
- 删除文档和知识库后不可召回。
- 多库检索不泄漏未选知识库。
- 非法拆分 JSON 回退。
- 超过 5 个子问题截断。
- 引用越权、数字不在证据、二次校验失败。
- 未验证正文不出现在 SSE。
- API、SSE 和日志不泄漏 Key、prompt 或路径。

Fake Provider 指标：

- Top 7 命中率不低于 85%。
- 知识库外问题拒答率 100%。
- 局部证据不足通过率 100%。
- 同库重复拦截率 100%。
- 未选知识库泄漏数量 0。
- 删除后召回数量 0。

## 17. 前端对接边界

前端只依赖 API DTO 和 SSE schema，不感知 SQLite、Chroma、Provider、prompt 或本地路径。上传走异步端点：提交后拿到 `job_id`，轮询 `GET /api/jobs/{job_id}` 并把逐文件 `progress` 映射到“校验 / 切块 / 向量化 / 保存 / 完成”阶段展示。

异步上传交互：

- 上传期间调用 `POST /api/jobs/{job_id}/cancel`，请求后继续轮询到 `cancelled` 或 `completed`。
- job 已并发完成导致 `409 upload_not_cancelable` 时按完成处理，不展示为取消失败。
- `cancelled` item 渲染为“已取消 / 未入库”，已完成 item 继续展示文档结果。
- 失败项保留浏览器内存中的 `File` 引用；“重新提交失败文件”只把匹配的失败文件再次提交到 async 端点，生成新 job，不在服务端保存重试状态。

前端状态：

```text
idle
→ decompose
→ retrieve
→ rerank
→ judge
→ generate
→ validate
→ regenerate?
→ compose
→ completed | refused | error
```

收到 `answer_final` 前不渲染正文，只渲染阶段状态和占位提示。

## 18. M1 实施清单

1. 创建 `pyproject.toml` 和 package metadata。
2. 实现 `Settings.from_env` 与配置校验。
3. 实现错误分类和安全日志工具。
4. 实现 SQLite migration。
5. 实现 uploads 路径安全。
6. 实现 Chroma collection registry 与兼容校验。
7. 实现 Provider Protocol 和 Online Provider 骨架。
8. 实现测试 Fake Provider。
9. 实现 `serve / doctor / rebuild / eval` CLI 入口。
10. 增加 M1 pytest。

完成标准：

- 空数据目录可初始化。
- 重复启动不重复建表。
- active collection 唯一。
- 模型或维度不一致时拒绝启动。
- API Key 缺失时状态正确。
- Fake Provider 测试全部通过。
- 不存在产品级 offline 模式或数据目录。

## 19. 课程集成

课程站位于 `web/`，继续使用 VitePress；`rag-app` 不嵌入课程站，课程站也不承载工作台业务。

集成点：

- 课程站提供 `rag-app` 工作台启动与验收页。
- 课程首页和导航提供工作台指南入口。
- 固定工作台入口为 `http://127.0.0.1:8010`。
- 课程页展示项目边界、启动命令、模型配置安全和人工验收清单。
- 课程站源码和构建产物不得包含 API Key、`.env` 内容、prompt、上传全文或本地数据。

验收方式：

1. 构建课程站。
2. 检查新增页面、导航和首页链接可访问。
3. 扫描课程源码和构建产物中的密钥形态，确认没有真实 Key。
4. 启动 `rag-app` 并按课程页清单完成基础状态、知识库、证据问答和生命周期检查。

## 20. 后续演进

- 网页抓取 / 浏览器采集与 OCR（当前只支持已上传的 PDF / Word / HTML 文件）。
- 异步 job 持久化（当前取消与前端重试已实现，但 job 仍仅存在于内存）。
- 场景路由与多个 workflow。
- 会话持久化与隐私清理。
- API 版本化、认证和限流。
- 更细粒度的检索评测（chunk 级相关性标注、多查询集与分领域基线）。
