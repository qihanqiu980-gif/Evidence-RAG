# M10 检索质量增强验证记录

日期：2026-10-06

## 验证范围

- 固定评测集新增 MRR、引用准确率、忠实度统计；忠实度使用独立 `faithfulness` judge。
- SQLite FTS5 trigram 关键词索引随入库 / 删除 / 启动自动维护；doctor 增加 `keyword_index` 对账。
- 关键词 BM25 与向量召回按 RRF 融合后统一 rerank；支持开关、两路权重与 `rrf_k` 配置。
- 关键词索引为派生索引，不改变业务 schema 版本。

## 自动化结果

```text
pytest               : 94 passed
ruff check src tests : passed
mypy src tests       : passed
frontend build       : passed
frontend e2e         : passed
doctor               : ok; 1 KB / 6 docs / 106 chunks / 106 vectors / 106 keyword_index
```

新增覆盖：关键词索引同步与自动重建、中英文关键词召回、知识库范围过滤、RRF 融合排序、
混合检索开关、权重与 RRF 配置校验、评测指标聚合与忠实度 judge。

## Online A/B 结果

同一数据面、同一 14 条评测集、`top_k=7`：

| 配置 | 通过 | 检索命中率 | MRR | 拒答通过率 | 引用准确率 | 忠实度 |
| --- | --- | --- | --- | --- | --- | --- |
| 纯向量（hybrid=false） | 14/14 | 100% | 0.8974 | 100% | 75.61% (31/41) | 100% (13/13) |
| 混合检索 第 1 轮 | 13/14 | 100% | 0.8974 | 92.86% | 75.00% | 100% |
| 混合检索 第 2 轮 | 12/14 | 100% | 0.8910 | 85.71% | 81.25% | 100% |
| 混合检索（充值后干净复跑） | 14/14 | 100% | 0.8974 | 100% | 81.08% (30/37) | 100% (15/15) |

两次混合运行中的失败 case 均满足：检索已命中（`retrieval=True` 且有 MRR）、
失败 case 自身 `failed_requests > 0`、未产生引用的 answer 来自 fail-closed 拒答。
其中第 1 轮唯一失败 `wifi-change-password` 是一次 provider 请求失败；第 2 轮末尾
`partial-mesh-and-price`、`followup-mesh-limit` 在 judge 阶段被供应商 403 拒绝，
随后用最小 chat 请求复核仍返回 403，确认是供应商侧故障 / 配额问题，与检索实现无关。
`wifi-change-password` 在第 2 轮中通过且引用 4/4。

结论：混合检索未观察到命中率、引用质量或忠实度回退；MRR 与纯向量基线持平
（差异来自在线 rerank 的运行间波动）。默认启用混合检索，保留
`RAG_APP_HYBRID_RETRIEVAL=false` 一键回退。

充值并确认 chat 恢复 200 后，完整 14 条混合评测复跑达到 14/14，且
`retries=0`、`failed_requests=0`、`faithfulness_errors=0`。该轮作为 M10 的干净记录。

## 2026-10-06 补跑前健康检查

尝试补跑完整混合评测前，先做了最小 provider 健康检查：

| Provider | 结果 |
| --- | --- |
| Embedding | 200 |
| Rerank | 200 |
| Chat | 403 |

Chat 的供应商错误为 `AllocationQuota.FreeTierOnly`：
`Free quota exhausted. To continue accessing the model on a paid basis, please add funds
or disable the "use free tier only" mode in the management console.`

因此本轮未继续消耗完整评测请求，也未产生新的质量基线。另有一次在本地沙箱内执行的
完整命令因 DNS 受限全部失败（0/14），该结果不代表产品或检索质量，不应归档。

## 2026-10-06 充值后复跑

供应商账户充值后，最小健康检查结果为 embedding/rerank/chat 全部 200。
第一次完整复跑出现 5 次间歇性 provider 快速失败（10/14），未作为质量基线；
最小接口复查恢复 200 并冷却 30 秒后重跑，最终结果：

```text
cases=14 passed=14
retrieval_hit_rate=100.00%
retrieval_mrr=0.8974
refusal_pass_rate=100.00%
citation_accuracy=81.08% (30/37)
faithfulness_rate=100.00% (15/15)
faithfulness_errors=0
provider_retries=0
provider_failed_requests=0
duration_ms_total=805743
tokens_total=627273
```

该结果确认 M10 混合检索在无基础设施故障条件下没有命中率、拒答或忠实度回退。

## 数据面

```text
doctor: ok
knowledge_bases=1
documents=6
chunks=106
vectors=106
keyword_index=106
```

M10 开工前备份：`backups-20261005-143419.tar.gz`（由 `rag-app backup backups` 生成，
不含 `.env` 或密钥）。
