# 本地 RAG 知识库工作台

`rag-app` 是一个本地数据、在线模型、证据约束的通用 RAG 知识库工作台。当前包含 M1-M8 能力：数据层、Markdown/纯文本/PDF/Word/HTML 多格式入库、向量索引、检索重排、证据判定、生成校验、问答 SSE、React 工作台、doctor、rebuild、备份恢复、固定评测、运行稳定性和课程站集成。

## 文档

- 产品需求：`docs/PRD.md`
- 技术设计：`docs/TECHNICAL_DESIGN.md`
- UI 设计：`docs/UI设计文档.md`

## 开发

```bash
cd rag-app
python3.12 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/rag-app doctor
```

启动本地服务（默认 `127.0.0.1:8010`）：

```bash
.venv/bin/rag-app serve
```

## 前端

```bash
cd frontend
pnpm install
pnpm run build
pnpm run test:e2e
```

构建产物输出到 `frontend/dist`，由 FastAPI 在 8010 同端口托管。开发前端时可另启 `pnpm run dev`，Vite 会把 `/api` 代理到 8010。
E2E 会临时启动 8011 端口的 FastAPI 测试实例，注入 Fake Provider 和独立数据目录，不会访问网络或污染 `data/`。

## 当前后端边界

产品运行时仅使用 online 模型；自动化测试使用注入的 Fake Provider，不访问网络或读取 `.env`。第一版固定执行知识证据问答 workflow：问题拆分、检索、重排、语义判定、生成、引用与数字校验、一次重生成、局部拒答，并通过 `POST /api/chat/stream` 输出 SSE。

## M5 评测与 online 验收

1. 将 `.env.example` 复制为 `.env`，填入有效的 `RAG_APP_API_KEY`。
2. 启动服务后，在页面创建知识库并导入 6 份示例 Markdown。
3. 运行数据面检查：

```bash
.venv/bin/rag-app doctor
```

4. 在已入库的知识库上运行固定评测：

```bash
.venv/bin/rag-app eval --dataset eval/cases.jsonl --top-k 7
```

评测集包含 14 条用例，覆盖规格、安装、WiFi、Mesh、恢复出厂、故障排查、保修、知识库外、复合问题、局部证据不足与多轮追问。Fake Provider 自动化指标在 `pytest` 中执行；CLI `eval` 只使用 online Provider，不提供产品级 fake 开关。

评测同时输出 case / 汇总延迟、p50 / p95、token usage、Provider 重试与失败统计。可在 `.env` 中选择性配置每 1K token 单价和 3 位货币代码；价格或供应商 usage 缺失时，`estimated_cost` 为 `null`，不会生成伪估算。

当前 Markdown online 基线记录见 `eval/baselines/2026-10-05-online.md`：14/14 通过，检索命中率 100%，拒答预期通过率 100%。文档 Adapter 扩展后该基线保留不变，用于对比扩展前后效果。


## M8 文档接入增强

支持 Markdown、纯文本 `.txt`、PDF、Word `.docx`、HTML 文档入库。所有格式先经统一 Adapter 抽取为纯文本，再走与 Markdown 相同的切块、指纹、向量化与证据检索流程；原始文件按原扩展名保存在 `uploads/`。

上传改为异步：`POST /api/knowledge-bases/{kb_id}/documents/async` 立即返回 `job_id`，`GET /api/jobs/{job_id}` 轮询任务进度与逐文件结果，工作台前端自动轮询并展示阶段进度。

数据安全网：`rag-app backup [目录]` 将 SQLite、uploads、Chroma 与 manifest 打包为 `.tar.gz` 快照；`rag-app restore <归档> --yes` 恢复当前数据目录。改上传/切块逻辑前建议先做一次备份。

## M6 运行稳定性

`rag-app serve` 会先初始化安全日志，并对已有数据目录执行启动前 doctor；错误阻断启动，警告继续启动。全新空目录会直接初始化，不会误判为索引缺失。Provider 对连接错误和 HTTP 5xx 使用可配置的指数退避重试，服务退出或启动失败时会幂等释放自持有的模型客户端。

## M7 课程集成

课程站提供 `rag-app` 工作台启动与验收指南，入口位于 `web/docs/rag/workbench/`。课程首页和导航链接到该指南，工作台固定入口仍是 `http://127.0.0.1:8010`。课程站只负责讲解和跳转，不嵌入 React 工作台，也不得包含 API Key 或 `.env` 内容。
