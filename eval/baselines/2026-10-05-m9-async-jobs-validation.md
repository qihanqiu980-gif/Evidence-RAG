# M9 异步任务健壮性验证记录

日期：2026-10-05

## 验证范围

- 服务启动时清理遗留 spool 文件，真实目录跳过，单个删除失败不阻断启动。
- 异步提交前校验文件名、扩展名、大小和数量；非法请求不创建 job。
- 新增 job 取消 API；覆盖排队取消、阶段边界取消和已完成项保留。
- 前端支持取消状态展示、取消请求和失败文件重新提交；重试总是创建新 job，服务端不保留重试状态。

## 自动化结果

```text
pytest             : 89 passed
ruff check src tests: passed
mypy src tests      : passed
frontend build      : passed
frontend e2e        : passed
doctor              : status ok; knowledge_bases=1 documents=6 chunks=106 vectors=106
data/spool          : empty
```

E2E 覆盖了失败文件按钮出现、点击后再次提交 async 端点，并确认第二次 job 仍按失败结果收敛。

## Online 基线复核

在 M9 开工前，rerank 恢复后已按用户要求完成：

- 严格知识库外拒答：通过，`refused=true`、无引用、SSE 正常完成。
- 原产品手册知识库 online 评测：14/14 通过，检索命中率 100%，拒答预期通过率 100%，Provider retry / failed request 均为 0。

M9 未修改切块、检索、重排和生成语义；本数据面 doctor 仍保持 M8 基线计数。
