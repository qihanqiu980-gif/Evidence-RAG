# 2026-10-05 M8 online validation

## Scope

- Document adapter: Markdown / plain text / PDF / Word / HTML ingestion path.
- Async upload: `POST /api/knowledge-bases/{kb_id}/documents/async` plus job polling.
- Evidence QA: real online Chat, Embedding, and Rerank providers.
- Regression baseline: the existing six-document Markdown knowledge base.
- Cleanup: restore the pre-validation data-plane snapshot after acceptance.

This record intentionally omits resume filenames, document content, and personal information.

## Real Document Upload

Two real resume files were used:

| Input format | Async result | Chunk count |
| --- | --- | ---: |
| DOCX | completed | 4 |
| PDF | completed | 4 |

Validation points:

- The async endpoint returned `202` with a `job_id` immediately.
- Both job items reached `completed`.
- Both items reported `校验`, `切块`, `向量化`, `保存`, and `完成`.
- Extracted text was readable for both formats.
- Uploads preserved the original `.docx` and `.pdf` extensions.
- The document list showed both records as `ready`.

## Evidence QA

Real online QA passed for resume-content questions:

- The workflow completed decompose, retrieve, rerank, judge, generate, validate, and compose.
- Answers used verified citation IDs.
- Quantified statements and education/ranking facts were supported by retrieved evidence.

Strict out-of-scope validation also passed:

```text
question type: completely external fact
refused: true
citations: []
reason: related material found, but insufficient direct evidence
SSE completion: completed
```

## Markdown Regression Baseline

Command:

```bash
rag-app eval --dataset eval/cases.jsonl --top-k 7 \
  --kb-id 9060e1a2-4580-4dfc-b3f8-113844ed5be5
```

Final result:

| Metric | Result |
| --- | ---: |
| Cases | 14 |
| Overall passed | 14 |
| Retrieval hit rate | 100.00% |
| Refusal expectation pass rate | 100.00% |
| Provider failed requests | 0 |
| Scope leaks | 0 |

This matches `2026-10-05-online.md`; the M8 document adapter and async upload changes did not regress the Markdown baseline.

One intermediate run reported 9/14 with five transient provider failures. A minimal Rerank request returned `200` after that run, and a full cooldown rerun produced the final 14/14 result above with zero provider failures. The intermediate run is not a quality baseline.

## Cleanup And Final Checks

The pre-validation data-plane snapshot was restored after acceptance.

```text
doctor: ok
knowledge_bases=1
documents=6
chunks=106
vectors=106
```

Frontend E2E passed after cleanup:

```text
frontend e2e passed
```

## Conclusion

M8 online validation passed. The local data plane was restored to the original Markdown baseline state and remains internally consistent.
