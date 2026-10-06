# M11 知识发现验证记录

日期：2026-10-06

## 结论

M11 代码实现完成。自动化测试、静态检查、前端构建和浏览器 E2E 均通过；SQLite live 数据面已从 schema v1 迁移到 v2，doctor 计数保持一致。真实 Provider 知识发现两次完整重试后均返回 `provider_unavailable`，因此线上模型生成的主题地图尚未取得成功样本，已保留失败状态并待供应商恢复后重新分析。

## 验证快照

```text
pytest -q                          : 96 passed
ruff check src tests               : passed
mypy src tests                     : passed
frontend build                     : passed
frontend e2e                       : passed
doctor                             : ok
sqlite schema                      : v2
```

doctor 数据面：

```text
knowledge_bases=1
documents=6
chunks=106
vectors=106
keyword_index=106
```

## E2E 覆盖

- 导入 6 份示例文档后自动生成知识地图。
- 页面展示主题“产品规格”、文档/chunk/主题/问题统计和来源绑定。
- 点击“星云智联 AX6000 支持 WiFi 6 吗？”后进入 `/api/chat/stream`。
- 等待回答完成，展开证据并核对 verified 答案与引用。
- 桌面与移动布局截图通过脚本内置检查。

## 真实 Provider 记录

- 第一次手动分析：`provider_unavailable / Provider is unreachable`。
- 第二次手动分析：同样失败。
- 本地服务无崩溃，SQLite 无主题写入，API 返回 failed 终态，页面可重新分析。
- 该结果不判定为 discovery 代码回归；供应商恢复后应先重新分析当前 KB，再抽查主题、问题和点击问答链路。

## 回滚

开工前备份：`rag-app/backups/pre-m11-discovery.tar.gz`

