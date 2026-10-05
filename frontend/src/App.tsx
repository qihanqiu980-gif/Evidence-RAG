import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Chat from './Chat'
import {
  ApiError,
  createKnowledgeBase,
  deleteDocument,
  deleteKnowledgeBase,
  getStatus,
  importDemoDocuments,
  listDocuments,
  listKnowledgeBases,
  uploadDocumentsAsync,
  getUploadJob,
  cancelUploadJob,
} from './api'
import { Badge, Icon, Modal, PageHeading } from './ui'
import type { DocumentSummary, KnowledgeBase, SystemStatus, UploadJobSummary } from './types'

type Page = 'overview' | 'knowledge' | 'chat'
type PendingDelete =
  | { kind: 'knowledge'; knowledgeBaseId: string }
  | { kind: 'document'; knowledgeBaseId: string; documentId: string }
type UploadOutcome = 'success' | 'duplicate' | 'failed' | 'cancelled'
type UploadEntry = {
  key: string
  filename: string
  outcome: UploadOutcome
  message: string
  progress: string[]
}

const navItems: { page: Page; label: string; icon: string }[] = [
  { page: 'overview', label: '概览', icon: 'grid' },
  { page: 'knowledge', label: '知识库', icon: 'book' },
  { page: 'chat', label: '问答', icon: 'chat' },
]

const serviceLabel: Record<
  'connecting' | 'ready' | 'configuration_missing' | 'storage_inconsistent' | 'error',
  { text: string; tone: 'blue' | 'neutral' | 'warn' | 'danger' }
> = {
  connecting: { text: '连接中', tone: 'neutral' },
  ready: { text: '已连接', tone: 'blue' },
  configuration_missing: { text: '配置错误', tone: 'warn' },
  storage_inconsistent: { text: '索引异常', tone: 'warn' },
  error: { text: '服务异常', tone: 'danger' },
}

const uploadStages = ['校验', '切块', '向量化', '保存', '完成']

const formatDateTime = (value: string) => {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('zh-CN', { hour12: false })
}

const formatBytes = (bytes: number) => `${Math.round(bytes / 1024 / 1024)} MB`

const batchEntries = (result: {
  documents: { filename: string; progress: string[] }[]
  errors: { filename: string; code: string; message: string }[]
}): UploadEntry[] => [
  ...result.documents.map((item) => ({
    key: `ok-${item.filename}-${crypto.randomUUID()}`,
    filename: item.filename,
    outcome: 'success' as const,
    message: '已完成入库',
    progress: item.progress,
  })),
  ...result.errors.map((item) => ({
    key: `error-${item.filename}-${crypto.randomUUID()}`,
    filename: item.filename,
    outcome:
      item.code === 'duplicate_document'
        ? ('duplicate' as const)
        : item.code === 'upload_cancelled'
          ? ('cancelled' as const)
          : ('failed' as const),
    message: item.message,
    progress: [],
  })),
]

const jobEntries = (job: UploadJobSummary): UploadEntry[] =>
  job.items.map((item) => ({
    key: `job-${item.filename}-${crypto.randomUUID()}`,
    filename: item.filename,
    outcome:
      item.status === 'completed'
        ? ('success' as const)
        : item.status === 'cancelled'
          ? ('cancelled' as const)
        : item.code === 'duplicate_document'
          ? ('duplicate' as const)
          : ('failed' as const),
    message:
      item.status === 'completed'
        ? '已完成入库'
        : item.status === 'failed'
          ? (item.message ?? '入库失败')
          : item.status === 'cancelled'
            ? (item.message ?? '已取消，未入库')
            : item.status === 'pending' || item.status === 'processing'
              ? '处理中'
              : '处理中',
    progress: item.progress,
  }))

function UploadStages({ progress, processing }: { progress: string[]; processing: boolean }) {
  return (
    <ol className="stage-list" aria-label="入库阶段">
      {uploadStages.map((stage) => {
        const state = processing
          ? 'idle'
          : progress.includes(stage)
            ? 'done'
            : 'idle'
        return (
          <li key={stage} className={processing ? 'idle pulse' : state}>
            {processing ? '○' : state === 'done' ? '✓' : '○'} {stage}
          </li>
        )
      })}
    </ol>
  )
}

export default function App() {
  const [page, setPage] = useState<Page>('overview')
  const [service, setService] = useState<
    'connecting' | 'ready' | 'configuration_missing' | 'storage_inconsistent' | 'error'
  >('connecting')
  const [status, setStatus] = useState<SystemStatus>()
  const [bases, setBases] = useState<KnowledgeBase[]>([])
  const [documents, setDocuments] = useState<DocumentSummary[]>([])
  const [activeId, setActiveId] = useState<string>()
  const [loadError, setLoadError] = useState<string>()
  const [creating, setCreating] = useState(false)
  const [createError, setCreateError] = useState<string>()
  const [newName, setNewName] = useState('')
  const [pendingDelete, setPendingDelete] = useState<PendingDelete>()
  const [deleting, setDeleting] = useState(false)
  const [deleteError, setDeleteError] = useState<string>()
  const [uploading, setUploading] = useState(false)
  const [uploadEntries, setUploadEntries] = useState<UploadEntry[]>()
  const [activeJobId, setActiveJobId] = useState<string>()
  const [cancelRequested, setCancelRequested] = useState(false)
  const [cancelError, setCancelError] = useState<string>()
  const fileRef = useRef<HTMLInputElement>(null)
  const activeIdRef = useRef<string | undefined>(undefined)
  const retryFilesRef = useRef<File[]>([])
  const retryKbIdRef = useRef<string | undefined>(undefined)

  activeIdRef.current = activeId

  const refresh = useCallback(async (showConnecting = false) => {
    if (showConnecting) setService('connecting')
    try {
      const [nextStatus, nextBases] = await Promise.all([getStatus(), listKnowledgeBases()])
      const nextDocuments = (
        await Promise.all(nextBases.map((base) => listDocuments(base.id)))
      ).flat()
      const selected =
        nextBases.find((base) => base.id === activeIdRef.current)?.id ?? nextBases[0]?.id
      setStatus(nextStatus)
      setBases(nextBases)
      setDocuments(nextDocuments)
      setActiveId(selected)
      setService(nextStatus.status)
      setLoadError(undefined)
    } catch (error) {
      const message = error instanceof ApiError ? error.message : '本地服务暂时不可用，请稍后重试。'
      setService('error')
      setLoadError(message)
    }
  }, [])

  useEffect(() => {
    void refresh(true)
  }, [refresh])

  const active = bases.find((base) => base.id === activeId)
  const activeDocuments = useMemo(
    () => documents.filter((document) => document.kb_id === active?.id),
    [documents, active?.id],
  )
  const operationsDisabled = service !== 'ready'
  const currentService = serviceLabel[service]
  const maxFiles = status?.limits.max_files_per_upload ?? 20
  const maxFileBytes = status?.limits.max_file_bytes ?? 5 * 1024 * 1024
  const acceptedExtensions = status?.limits.accepted_extensions ?? ['.md']

  const runBatch = async (
    entries: Pick<UploadEntry, 'filename' | 'outcome' | 'message'>[],
    action: () => Promise<{ documents: { filename: string; progress: string[] }[]; errors: { filename: string; code: string; message: string }[] }>,
  ) => {
    setUploading(true)
    setActiveJobId(undefined)
    setCancelRequested(false)
    setCancelError(undefined)
    setUploadEntries(
      entries.length > 0
        ? entries.map((item) => ({
            ...item,
            key: `local-${item.filename}-${crypto.randomUUID()}`,
            progress: [],
          }))
        : [
            {
              key: `batch-${crypto.randomUUID()}`,
              filename: '示例资料',
              outcome: 'failed' as const,
              message: '已提交服务端，等待返回最终入库结果',
              progress: [],
            },
          ],
    )

    try {
      const result = await action()
      setUploadEntries(batchEntries(result))
      await refresh()
    } catch (error) {
      const message = error instanceof ApiError ? error.message : '本次入库未完成，请稍后重试。'
      setUploadEntries((current) =>
        (current ?? []).map((item) =>
          item.message.startsWith('已提交')
            ? { ...item, outcome: 'failed', message }
            : item,
        ),
      )
    } finally {
      setUploading(false)
      setActiveJobId(undefined)
      setCancelRequested(false)
    }
  }

  const submitFiles = async (valid: File[], invalid: UploadEntry[]) => {
    if (!active || uploading || operationsDisabled || valid.length === 0) return
    retryFilesRef.current = valid
    retryKbIdRef.current = active.id
    const localEntries = valid.map((file) => ({
      filename: file.name,
      outcome: 'failed' as const,
      message: '已提交，等待服务端返回最终入库结果',
    }))

    await runBatch([...localEntries, ...invalid], async () => {
      const { job_id: jobId } = await uploadDocumentsAsync(active.id, valid)
      setActiveJobId(jobId)

      let job: UploadJobSummary | undefined
      const deadline = Date.now() + 60_000
      while (Date.now() < deadline) {
        job = await getUploadJob(jobId)
        if (job.status === 'completed' || job.status === 'cancelled') break
        setUploadEntries(jobEntries(job))
        await new Promise((resolve) => setTimeout(resolve, 500))
      }
      if (!job || (job.status !== 'completed' && job.status !== 'cancelled')) {
        throw new ApiError('timeout', '入库超时，请稍后刷新查看文档状态。')
      }
      return {
        documents: job.items
          .filter((item) => item.status === 'completed')
          .map((item) => ({ filename: item.filename, progress: item.progress })),
        errors: job.items
          .filter((item) => item.status !== 'completed')
          .map((item) => ({
            filename: item.filename,
            code: item.code ?? (item.status === 'cancelled' ? 'upload_cancelled' : 'failed'),
            message:
              item.message ??
              (item.status === 'cancelled' ? '已取消，未入库' : '入库失败'),
          })),
      }
    })
  }

  const onPickFiles = async (files: FileList | null) => {
    if (!files || !active || uploading || operationsDisabled) return
    const selected = Array.from(files)
    if (fileRef.current) fileRef.current.value = ''

    if (selected.length > maxFiles) {
      setUploadEntries([
        {
          key: `limit-${crypto.randomUUID()}`,
          filename: '本次选择的文件',
          outcome: 'failed',
          message: `单次最多上传 ${maxFiles} 个文件，请分批上传。`,
          progress: [],
        },
      ])
      return
    }

    const valid: File[] = []
    const invalid: { filename: string; outcome: 'failed'; message: string }[] = []
    for (const file of selected) {
      const lower = file.name.toLowerCase()
      if (!acceptedExtensions.some((ext) => lower.endsWith(ext.toLowerCase()))) {
        invalid.push({ filename: file.name, outcome: 'failed', message: '不支持的文件类型。' })
        continue
      }
      if (file.size > maxFileBytes) {
        invalid.push({
          filename: file.name,
          outcome: 'failed',
          message: `文件超过 ${formatBytes(maxFileBytes)} 限制。`,
        })
        continue
      }
      if (lower.endsWith('.md') || lower.endsWith('.txt') || lower.endsWith('.html') || lower.endsWith('.htm')) {
        try {
          const text = new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer())
          if (!text.trim()) {
            invalid.push({ filename: file.name, outcome: 'failed', message: '文件内容为空。' })
            continue
          }
        } catch {
          invalid.push({ filename: file.name, outcome: 'failed', message: '文件不是有效的 UTF-8 编码。' })
          continue
        }
      }
      valid.push(file)
    }

    if (valid.length === 0) {
      setUploadEntries(
        invalid.map((item) => ({ ...item, key: `invalid-${crypto.randomUUID()}`, progress: [] })),
      )
      return
    }

    await submitFiles(
      valid,
      invalid.map((item) => ({ ...item, key: `invalid-${crypto.randomUUID()}`, progress: [] })),
    )
  }

  const cancelJob = async () => {
    if (!activeJobId || cancelRequested) return
    try {
      await cancelUploadJob(activeJobId)
      setCancelRequested(true)
      setCancelError(undefined)
    } catch (error) {
      if (error instanceof ApiError && error.code === 'upload_not_cancelable') {
        setCancelRequested(true)
        return
      }
      setCancelError(error instanceof ApiError ? error.message : '取消请求失败，请稍后重试。')
    }
  }

  const retryFailedFiles = async () => {
    if (!active || uploading || operationsDisabled) return
    const failedNames = new Set(
      (uploadEntries ?? [])
        .filter((entry) => entry.outcome === 'failed')
        .map((entry) => entry.filename),
    )
    const files = retryFilesRef.current.filter((file) => failedNames.has(file.name))
    if (files.length === 0 || retryKbIdRef.current !== active.id) return
    await submitFiles(files, [])
  }

  const create = async () => {
    const name = newName.trim()
    if (!name) return
    setCreating(false)
    try {
      const created = await createKnowledgeBase(name)
      setActiveId(created.id)
      activeIdRef.current = created.id
      setNewName('')
      setCreateError(undefined)
      setPage('knowledge')
      await refresh()
    } catch (error) {
      setCreateError(error instanceof ApiError ? error.message : '创建失败，请稍后重试。')
    }
  }

  const confirmDelete = async () => {
    if (!pendingDelete || deleting) return
    setDeleting(true)
    setDeleteError(undefined)
    try {
      if (pendingDelete.kind === 'knowledge') {
        await deleteKnowledgeBase(pendingDelete.knowledgeBaseId)
        if (activeIdRef.current === pendingDelete.knowledgeBaseId) activeIdRef.current = undefined
      } else {
        await deleteDocument(pendingDelete.knowledgeBaseId, pendingDelete.documentId)
      }
      setPendingDelete(undefined)
      await refresh()
    } catch (error) {
      setDeleteError(error instanceof ApiError ? error.message : '删除失败，请稍后重试。')
    } finally {
      setDeleting(false)
    }
  }

  const brand = (
    <div className="w-full min-w-0 lg:w-auto lg:block">
      <div className="break-words text-[20px] font-semibold leading-tight lg:text-[22px]">
        本地 RAG 知识库工作台
      </div>
      <div className="mt-0.5 text-[13px] text-[var(--muted)]">
        知识入库 · 检索 · 证据 · 生成与校验
      </div>
    </div>
  )

  const nav = (
    <nav aria-label="主导航" className="flex gap-1 lg:flex-col">
      {navItems.map((item) => (
        <button
          key={item.page}
          type="button"
          className="nav-item lg:w-full"
          aria-current={page === item.page ? 'page' : undefined}
          onClick={() => setPage(item.page)}
        >
          <Icon name={item.icon} size={18} />
          {item.label}
        </button>
      ))}
    </nav>
  )

  const pendingKnowledgeBase = bases.find(
    (base) => pendingDelete?.kind === 'knowledge' && base.id === pendingDelete.knowledgeBaseId,
  )
  const failedUploadNames = new Set(
    (uploadEntries ?? [])
      .filter((entry) => entry.outcome === 'failed')
      .map((entry) => entry.filename),
  )
  const retryableFiles =
    active && retryKbIdRef.current === active.id
      ? retryFilesRef.current.filter((file) => failedUploadNames.has(file.name))
      : []
  const pendingDocument =
    pendingDelete?.kind === 'document'
      ? documents.find((document) => document.id === pendingDelete.documentId)
      : undefined

  return (
    <div className="app-shell lg:grid lg:grid-cols-[216px_minmax(0,1fr)]">
      <aside className="sticky top-0 hidden h-screen flex-col border-r border-[var(--line)] bg-white px-4 py-6 lg:flex">
        <div className="px-3 pb-6">{brand}</div>
        {nav}
        <div className="mt-auto border-t border-[var(--line)] px-3 pt-4" role="status">
          <Badge tone={currentService.tone} busy={service === 'connecting'}>
            {currentService.text}
          </Badge>
        </div>
      </aside>

      <header className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 border-b border-[var(--line)] bg-white px-4 py-3 lg:hidden">
        {brand}
        <div className="flex w-full flex-wrap items-center justify-between gap-2">
          {nav}
          <div role="status">
            <Badge tone={currentService.tone} busy={service === 'connecting'}>
              {currentService.text}
            </Badge>
          </div>
        </div>
      </header>

      <div className="flex min-h-screen min-w-0 flex-col lg:min-h-0">
        <main className="mx-auto w-full max-w-[1200px] flex-1 px-4 py-6 sm:px-6 sm:py-8 lg:px-10">
          {service !== 'ready' && (
            <div className="card mb-5 flex flex-wrap items-center justify-between gap-3 p-4">
              <div className="min-w-0">
                <div className="text-[15px] font-medium">{currentService.text}</div>
                <p className="m-0 break-words text-[13px] text-[var(--muted)]">
                  {loadError ??
                    (service === 'connecting'
                      ? '正在连接本地服务。'
                      : service === 'configuration_missing'
                        ? '模型服务配置不完整，入库和问答已暂停。'
                        : service === 'storage_inconsistent'
                          ? '本地数据一致性异常，请先运行 doctor 检查。'
                          : '本地服务暂时不可用。')}
                </p>
              </div>
              <button
                type="button"
                className="btn w-full sm:w-auto"
                onClick={() => void refresh(true)}
                disabled={service === 'connecting'}
              >
                <Icon name="refresh" size={16} />
                重试连接
              </button>
            </div>
          )}

          {page === 'overview' && (
            <>
              <PageHeading title="概览" sub="查看知识库存储情况并进入常用操作。">
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => setPage('chat')}
                  disabled={bases.length === 0 || operationsDisabled}
                >
                  进入问答
                </button>
              </PageHeading>

              <h2 className="t-section m-0 mb-3">存储统计</h2>
              <div className="grid gap-4 md:grid-cols-3">
                {[
                  {
                    label: '知识库数量',
                    value: status?.knowledge_base_count ?? bases.length,
                    unit: '个',
                    note: '独立检索范围',
                    icon: 'book',
                  },
                  {
                    label: '文档数量',
                    value: status?.document_count ?? documents.length,
                    unit: '份',
                    note: '已入库文档',
                    icon: 'file',
                  },
                  {
                    label: '文本块数量',
                    value: status?.chunk_count ?? bases.reduce((sum, item) => sum + item.chunk_count, 0),
                    unit: '个',
                    note: '已完成切块的知识片段',
                    icon: 'layers',
                  },
                ].map((item) => (
                  <div key={item.label} className="card p-5">
                    <div className="flex items-center justify-between text-[14px] text-[var(--muted)]">
                      {item.label}
                      <Icon name={item.icon} size={16} />
                    </div>
                    <div className="mt-3 flex items-baseline gap-1.5">
                      <span className="text-[36px] font-semibold leading-none tabular-nums">
                        {item.value}
                      </span>
                      <span className="text-[15px] text-[var(--muted)]">{item.unit}</span>
                    </div>
                    <p className="m-0 mt-3 text-[13px] text-[var(--muted)]">{item.note}</p>
                  </div>
                ))}
              </div>
              {bases.length === 0 && (
                <p className="m-0 mt-3 text-[14px] text-[var(--muted)]">
                  还没有知识库，创建后即可上传文档。
                </p>
              )}

              <div className="card mt-6 flex flex-wrap items-center justify-between gap-4 p-5">
                <div className="min-w-0">
                  <h2 className="t-section m-0">知识库管理</h2>
                  <p className="m-0 mt-1 text-[15px] text-[var(--muted)]">
                    创建知识库、上传文档、查看文档状态与入库结果。
                  </p>
                </div>
                <button type="button" className="btn" onClick={() => setPage('knowledge')}>
                  管理知识库
                </button>
              </div>

              <h2 className="t-section m-0 mb-3 mt-8">入库规则</h2>
              <ul className="card m-0 list-none divide-y divide-[var(--line)] p-0">
                {[
                  '支持 Markdown、纯文本、PDF、Word、HTML 文档。',
                  `单文件不超过 ${formatBytes(maxFileBytes)}，单次最多 ${maxFiles} 个文件。`,
                  '同一知识库内重复内容会在向量化前拦截。',
                ].map((rule) => (
                  <li key={rule} className="flex items-center gap-3 px-5 py-3 text-[15px]">
                    <span className="text-[var(--muted)]">
                      <Icon name="check" size={16} />
                    </span>
                    {rule}
                  </li>
                ))}
              </ul>
            </>
          )}

          {page === 'knowledge' && (
            <>
              <PageHeading title="知识库" sub="管理知识库与文档。">
                <button type="button" className="btn btn-primary" onClick={() => setCreating(true)}>
                  <Icon name="plus" size={16} />
                  新建知识库
                </button>
              </PageHeading>

              {bases.length === 0 ? (
                <div className="card flex flex-col items-center px-6 py-16 text-center">
                  <h2 className="t-section m-0">还没有知识库</h2>
                  <p className="m-0 mt-1 max-w-[380px] text-[15px] text-[var(--muted)]">
                    知识库是独立的检索范围。创建后可以上传文档并在问答中使用。
                  </p>
                  <button
                    type="button"
                    className="btn btn-primary mt-5"
                    onClick={() => setCreating(true)}
                  >
                    <Icon name="plus" size={16} />
                    新建知识库
                  </button>
                </div>
              ) : (
                <div className="grid items-start gap-5 xl:grid-cols-[280px_minmax(0,1fr)]">
                  <ul
                    className="m-0 grid list-none gap-2 p-0 sm:grid-cols-2 xl:grid-cols-1"
                    aria-label="知识库列表"
                  >
                    {bases.map((base) => {
                      const on = base.id === active?.id
                      return (
                        <li key={base.id}>
                          <button
                            type="button"
                            className={`card block w-full p-4 text-left ${
                              on
                                ? '!border-[var(--primary)] bg-[var(--primary-soft)]'
                                : 'hover:border-[var(--line-hover)]'
                            }`}
                            aria-current={on ? 'true' : undefined}
                            onClick={() => {
                              setActiveId(base.id)
                              setUploadEntries(undefined)
                            }}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <span className="min-w-0 break-all text-[15px] font-semibold">
                                {base.name}
                              </span>
                              <Badge tone="ok">已就绪</Badge>
                            </div>
                            <div className="mt-1.5 text-left text-[13px] text-[var(--muted)]">
                              {base.document_count} 份文档 · {base.chunk_count} 个文本块
                            </div>
                          </button>
                        </li>
                      )
                    })}
                  </ul>

                  {active && (
                    <section className="card min-w-0 overflow-hidden" aria-label="知识库详情">
                      <div className="flex flex-wrap items-start justify-between gap-4 p-4 sm:p-5">
                        <div className="min-w-0">
                          <div className="flex flex-wrap items-center gap-3">
                            <h2 className="t-section m-0 break-all">{active.name}</h2>
                            <Badge tone="ok">已就绪</Badge>
                          </div>
                          <p className="m-0 mt-1 text-[14px] text-[var(--muted)]">
                            {active.document_count} 份文档 · {active.chunk_count} 个文本块
                          </p>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <button
                            type="button"
                            className="btn"
                            onClick={() =>
                              void runBatch([], () => importDemoDocuments(active.id))
                            }
                            disabled={uploading || operationsDisabled}
                          >
                            导入示例资料
                          </button>
                          <button
                            type="button"
                            className="btn"
                            onClick={() => fileRef.current?.click()}
                            disabled={uploading || operationsDisabled}
                          >
                            <Icon name="upload" size={16} />
                            上传文档
                          </button>
                          {activeJobId && (
                            <button
                              type="button"
                              className="btn btn-danger"
                              onClick={() => void cancelJob()}
                              disabled={!uploading || cancelRequested}
                            >
                              <Icon name="close" size={16} />
                              {cancelRequested ? '取消中…' : '取消上传'}
                            </button>
                          )}
                          <button
                            type="button"
                            className="btn btn-danger"
                            onClick={() =>
                              setPendingDelete({ kind: 'knowledge', knowledgeBaseId: active.id })
                            }
                            disabled={uploading}
                          >
                            <Icon name="trash" size={16} />
                            删除知识库
                          </button>
                          <input
                            ref={fileRef}
                            type="file"
                            multiple
                            accept=".md,.txt,.pdf,.docx,.html,.htm"
                            className="sr-only"
                            tabIndex={-1}
                            aria-label="选择文档文件"
                            onChange={(event) => void onPickFiles(event.target.files)}
                          />
                        </div>
                      </div>
                      <p className="m-0 border-t border-[var(--line)] bg-[var(--surface-muted)] px-4 py-2.5 text-[13px] text-[var(--muted)] sm:px-5">
                        支持 .md / .txt / .pdf / .docx / .html · 单文件不超过 {formatBytes(maxFileBytes)} · 单次最多{' '}
                        {maxFiles} 个文件
                      </p>

                      {uploadEntries && (
                        <div className="border-t border-[var(--line)]">
                          <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-5">
                            <div className="min-w-0">
                              <h3 className="m-0 text-[15px] font-semibold">入库结果</h3>
                              <p className="m-0 text-[13px] text-[var(--muted)]">
                                共 {uploadEntries.length} 项，成功{' '}
                                {uploadEntries.filter((item) => item.outcome === 'success').length}
                                ，未入库{' '}
                                {uploadEntries.filter((item) => item.outcome !== 'success').length}
                              </p>
                              {cancelError && (
                                <p className="m-0 mt-1 text-[13px] text-[var(--danger)]">
                                  {cancelError}
                                </p>
                              )}
                            </div>
                            <div className="flex flex-wrap items-center gap-2">
                              {!uploading && retryableFiles.length > 0 && (
                                <button
                                  type="button"
                                  className="btn"
                                  onClick={() => void retryFailedFiles()}
                                >
                                  <Icon name="refresh" size={16} />
                                  重新提交失败文件
                                </button>
                              )}
                              <button
                                type="button"
                                className="icon-btn"
                                aria-label="关闭入库结果"
                                title="关闭入库结果"
                                onClick={() => {
                                  setUploadEntries(undefined)
                                  retryFilesRef.current = []
                                  retryKbIdRef.current = undefined
                                  setCancelError(undefined)
                                }}
                              >
                                <Icon name="close" size={16} />
                              </button>
                            </div>
                          </div>
                          <ul className="m-0 list-none p-0">
                            {uploadEntries.map((entry) => {
                              const processing = uploading && entry.outcome === 'failed'
                              const tone =
                                entry.outcome === 'success'
                                  ? ('ok' as const)
                                  : entry.outcome === 'duplicate'
                                    ? ('warn' as const)
                                    : entry.outcome === 'cancelled'
                                      ? ('neutral' as const)
                                    : processing
                                      ? ('neutral' as const)
                                      : ('danger' as const)
                              const text =
                                entry.outcome === 'success'
                                  ? '已就绪'
                                  : entry.outcome === 'duplicate'
                                    ? '重复文档'
                                    : entry.outcome === 'cancelled'
                                      ? '已取消'
                                    : processing
                                      ? '处理中'
                                      : '未入库'
                              return (
                                <li
                                  key={entry.key}
                                  className="flex flex-wrap items-start justify-between gap-3 border-t border-[var(--line)] px-4 py-3 sm:px-5"
                                >
                                  <div className="min-w-0 flex-1">
                                    <div className="break-all text-[14px] font-medium">
                                      {entry.filename}
                                    </div>
                                    {processing ? (
                                      <UploadStages progress={entry.progress} processing />
                                    ) : (
                                      <div className="text-[13px] text-[var(--muted)]">
                                        {entry.message}
                                      </div>
                                    )}
                                  </div>
                                  <Badge tone={tone} busy={processing}>
                                    {text}
                                  </Badge>
                                </li>
                              )
                            })}
                          </ul>
                        </div>
                      )}

                      {activeDocuments.length === 0 ? (
                        <div className="border-t border-[var(--line)] px-6 py-14 text-center">
                          <h3 className="m-0 text-[17px] font-semibold">这个知识库还没有文档</h3>
                          <p className="m-0 mt-1 text-[15px] text-[var(--muted)]">
                            上传文档文件，或导入示例资料开始。
                          </p>
                          <div className="mt-5 flex flex-wrap justify-center gap-2">
                            <button
                              type="button"
                              className="btn btn-primary"
                              onClick={() => fileRef.current?.click()}
                              disabled={uploading || operationsDisabled}
                            >
                              <Icon name="upload" size={16} />
                              上传文档
                            </button>
                            <button
                              type="button"
                              className="btn"
                              onClick={() => void runBatch([], () => importDemoDocuments(active.id))}
                              disabled={uploading || operationsDisabled}
                            >
                              导入示例资料
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div className="overflow-x-auto border-t border-[var(--line)]">
                          <table className="w-full min-w-[680px] border-collapse">
                            <thead>
                              <tr>
                                <th className="th">文档名称</th>
                                <th className="th">状态</th>
                                <th className="th text-right">文本块</th>
                                <th className="th">入库时间</th>
                                <th className="th text-right">操作</th>
                              </tr>
                            </thead>
                            <tbody>
                              {activeDocuments.map((document) => (
                                <tr key={document.id} className="hover:bg-[var(--surface-muted)]">
                                  <td className="td min-w-[220px]">
                                    <div className="flex items-start gap-2">
                                      <span className="mt-0.5 text-[var(--muted)]">
                                        <Icon name="file" size={16} />
                                      </span>
                                      <span className="break-all font-medium">{document.filename}</span>
                                    </div>
                                  </td>
                                  <td className="td">
                                    <Badge tone="ok">已就绪</Badge>
                                  </td>
                                  <td className="td text-right tabular-nums">{document.chunk_count}</td>
                                  <td className="td whitespace-nowrap text-[var(--muted)] tabular-nums">
                                    {formatDateTime(document.created_at)}
                                  </td>
                                  <td className="td text-right">
                                    <button
                                      type="button"
                                      className="icon-btn danger"
                                      aria-label={`删除文档 ${document.filename}`}
                                      title="删除文档"
                                      disabled={uploading}
                                      onClick={() =>
                                        setPendingDelete({
                                          kind: 'document',
                                          knowledgeBaseId: document.kb_id,
                                          documentId: document.id,
                                        })
                                      }
                                    >
                                      <Icon name="trash" size={16} />
                                    </button>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </section>
                  )}
                </div>
              )}
            </>
          )}

          <Chat
            bases={bases}
            documents={documents}
            active={page === 'chat'}
            disabled={operationsDisabled}
            goKnowledge={() => setPage('knowledge')}
          />
        </main>

        <footer className="mx-auto flex w-full max-w-[1200px] flex-wrap justify-between gap-2 border-t border-[var(--line)] px-4 py-4 text-[13px] text-[var(--muted)] sm:px-6 lg:px-10">
          <span>本地 RAG 知识库工作台</span>
          <span>本地数据 · 证据约束问答</span>
        </footer>
      </div>

      {creating && (
        <Modal
          title="新建知识库"
          onClose={() => setCreating(false)}
          actions={
            <>
              <button type="button" className="btn" onClick={() => setCreating(false)}>
                取消
              </button>
              <button
                type="button"
                id="create-knowledge-base"
                className="btn btn-primary"
                onClick={() => void create()}
                disabled={!newName.trim()}
              >
                创建
              </button>
            </>
          }
        >
          <label htmlFor="kb-name" className="mb-1.5 block text-[14px]">
            知识库名称
          </label>
          <input
            id="kb-name"
            className="field"
            autoFocus
            maxLength={100}
            placeholder="例如：产品资料"
            value={newName}
            onChange={(event) => setNewName(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') void create()
            }}
          />
          {createError && <p className="mt-2 text-[13px] text-[var(--danger)]">{createError}</p>}
        </Modal>
      )}

      {pendingDelete && (pendingKnowledgeBase || pendingDocument) && (
        <Modal
          title={pendingDelete.kind === 'knowledge' ? '删除知识库' : '删除文档'}
          onClose={() => setPendingDelete(undefined)}
          actions={
            <>
              <button type="button" className="btn" onClick={() => setPendingDelete(undefined)}>
                取消
              </button>
              <button
                type="button"
                className="btn btn-danger-solid"
                onClick={() => void confirmDelete()}
                disabled={deleting}
              >
                确认删除
              </button>
            </>
          }
        >
          <p className="m-0 mb-3 break-all rounded-[6px] bg-[var(--neutral-soft)] px-3 py-2 text-[15px] font-medium">
            {pendingDelete.kind === 'knowledge'
              ? pendingKnowledgeBase?.name
              : pendingDocument?.filename}
          </p>
          <p className="m-0 text-[15px]">
            {pendingDelete.kind === 'knowledge'
              ? `将删除该知识库下 ${pendingKnowledgeBase?.document_count ?? 0} 份文档的原始文件、元数据和向量索引，删除后无法恢复。`
              : '将删除该文档的原始文件、元数据和向量索引，删除后无法恢复，相关内容不再参与检索。'}
          </p>
          {deleteError && <p className="mt-3 text-[13px] text-[var(--danger)]">{deleteError}</p>}
        </Modal>
      )}
    </div>
  )
}
