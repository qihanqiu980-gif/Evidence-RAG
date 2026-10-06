import { useEffect, useMemo, useRef, useState } from 'react'
import { ApiError, streamChat } from './api'
import { Badge, Icon, PageHeading } from './ui'
import type {
  ChatHistoryMessage,
  DiscoverySummary,
  DiscoveryTopic,
  DocumentSummary,
  Evidence,
  EvidenceDecision,
  FinalAnswer,
  KnowledgeBase,
  SseEvent,
  WorkflowStage,
} from './types'

type TraceStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped'
type TraceItem = { stage: WorkflowStage; label: string; status: TraceStatus; message: string }
type AssistantMessage = {
  id: number
  role: 'assistant'
  question: string
  kbIds: string[]
  historySnapshot: { role: 'user' | 'assistant'; content: string }[]
  running: boolean
  done: boolean
  trace: TraceItem[]
  evidence: Evidence[]
  decisions: EvidenceDecision[]
  final?: FinalAnswer
  errorMessage?: string
  evidenceOpen: boolean
  traceOpen: boolean
}
type ChatMessage =
  | { id: number; role: 'user'; text: string }
  | AssistantMessage
type ScopedDiscoveryTopic = DiscoveryTopic & { knowledgeBaseName: string }

const stageLabels: { stage: WorkflowStage; label: string }[] = [
  { stage: 'decompose', label: '问题拆分' },
  { stage: 'retrieve', label: '检索' },
  { stage: 'rerank', label: '重排' },
  { stage: 'judge', label: '证据判定' },
  { stage: 'generate', label: '生成' },
  { stage: 'validate', label: '校验' },
  { stage: 'regenerate', label: '重新生成' },
  { stage: 'compose', label: '合成回答' },
]

const traceStatusLabel: Record<TraceStatus, string> = {
  pending: '等待',
  running: '进行中',
  completed: '已完成',
  failed: '未通过',
  skipped: '未触发',
}

const initialTrace = (): TraceItem[] =>
  stageLabels.map((item) => ({ ...item, status: 'pending', message: '' }))

const formatHeading = (path: string[]) =>
  path.length > 0 ? path.join(' > ') : '未记录标题路径'

function TraceList({ trace }: { trace: TraceItem[] }) {
  return (
    <ol className="m-0 list-none p-0">
      {trace.map((item, index) => (
        <li key={item.stage} className="relative flex gap-3 pb-3 last:pb-0">
          {index < trace.length - 1 && (
            <span
              className="absolute bottom-0 left-[9px] top-6 w-px bg-[var(--line)]"
              aria-hidden="true"
            />
          )}
          <span
            className={`z-[1] mt-0.5 grid h-5 w-5 shrink-0 place-items-center rounded-full border text-[var(--muted)] ${
              item.status === 'completed'
                ? 'border-[var(--ok)] bg-[var(--ok-soft)] text-[var(--ok)]'
                : item.status === 'failed'
                  ? 'border-[var(--danger)] bg-[var(--danger-soft)] text-[var(--danger)]'
                  : item.status === 'running'
                    ? 'pulse border-[var(--primary)] bg-[var(--primary-soft)] text-[var(--primary)]'
                    : 'border-[var(--line)] bg-white'
            }`}
          >
            {item.status === 'completed' && <Icon name="check" size={12} />}
            {item.status === 'failed' && <Icon name="close" size={12} />}
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-[14px] font-medium">{item.label}</span>
              <span
                className={`text-[13px] ${
                  item.status === 'failed' ? 'text-[var(--danger)]' : 'text-[var(--muted)]'
                }`}
              >
                {traceStatusLabel[item.status]}
              </span>
            </div>
            {item.status !== 'pending' && item.message && (
              <p className="m-0 break-words text-[13px] text-[var(--muted)]">{item.message}</p>
            )}
          </div>
        </li>
      ))}
    </ol>
  )
}

function EvidenceList({ evidence }: { evidence: Evidence[] }) {
  if (evidence.length === 0) {
    return <p className="m-0 text-[14px] text-[var(--muted)]">没有检索到相关资料。</p>
  }

  return (
    <ul className="m-0 grid list-none gap-3 p-0">
      {evidence.map((item) => (
        <li
          key={`${item.subquestion_id}-${item.reference_id}`}
          className="rounded-[6px] border border-[var(--line)] bg-white p-3.5"
        >
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div className="flex min-w-0 items-start gap-2">
              <span className="grid h-6 min-w-6 place-items-center rounded-[4px] bg-[var(--neutral-soft)] px-1 text-[13px] font-semibold">
                [{item.reference_id}]
              </span>
              <span className="min-w-0 break-all text-[14px] font-medium">
                {item.document_name}
              </span>
            </div>
            <Badge tone={item.support_status === 'supporting' ? 'ok' : 'neutral'}>
              {item.support_status === 'supporting'
                ? '回答证据'
                : '相关资料，不足以直接回答'}
            </Badge>
          </div>
          <dl className="m-0 mt-2.5 grid gap-x-3 gap-y-1 text-[13px] sm:grid-cols-[72px_1fr]">
            <dt className="text-[var(--muted)]">标题路径</dt>
            <dd className="m-0 break-words">{formatHeading(item.heading_path)}</dd>
            <dt className="text-[var(--muted)]">重排分数</dt>
            <dd className="m-0 tabular-nums">{item.rerank_score.toFixed(2)}</dd>
            <dt className="text-[var(--muted)]">子问题</dt>
            <dd className="m-0 break-words">{item.subquestion}</dd>
          </dl>
          <blockquote className="m-0 mt-2.5 break-words border-l-2 border-[#c6d3e8] bg-[#f7f8fa] px-3 py-2 text-[14px] leading-relaxed text-[#3b434d]">
            {item.content}
          </blockquote>
        </li>
      ))}
    </ul>
  )
}

function ExecutionPanel({ message }: { message?: AssistantMessage }) {
  const supportingCount =
    message?.evidence.filter((item) => item.support_status === 'supporting').length ?? 0
  const relatedCount =
    message?.evidence.filter((item) => item.support_status !== 'supporting').length ?? 0
  const answerableCount = message?.decisions.filter((item) => item.answerable).length ?? 0
  const state = message
    ? message.running
      ? { label: '处理中', tone: 'neutral' as const, busy: true }
      : message.errorMessage
        ? { label: '处理失败', tone: 'danger' as const, busy: false }
        : message.final?.refused
          ? { label: '全部拒答', tone: 'neutral' as const, busy: false }
          : message.final?.parts.some((part) => part.status === 'refused')
            ? { label: '局部拒答', tone: 'warn' as const, busy: false }
            : { label: '回答完成', tone: 'ok' as const, busy: false }
    : undefined

  return (
    <aside
      className="card min-h-[320px] self-start overflow-hidden xl:sticky xl:top-6 xl:min-h-[560px]"
      aria-label="执行过程"
    >
      <div className="flex items-center justify-between gap-3 border-b border-[var(--line)] px-4 py-3">
        <h2 className="m-0 text-[15px] font-semibold">执行过程</h2>
        {state ? (
          <Badge tone={state.tone} busy={state.busy}>
            {state.label}
          </Badge>
        ) : (
          <span className="text-[13px] text-[var(--muted)]">本次会话</span>
        )}
      </div>

      {!message ? (
        <div className="flex min-h-[240px] flex-col items-center justify-center px-6 py-10 text-center">
          <span className="text-[var(--muted)]">
            <Icon name="chat" size={34} />
          </span>
          <h3 className="m-0 mt-3 text-[16px] font-semibold">等待提问</h3>
          <p className="m-0 mt-1 text-[14px] text-[var(--muted)]">
            回答过程与引用证据会显示在这里。
          </p>
        </div>
      ) : (
        <div className="px-4 py-4">
          <div className="text-[13px] font-medium text-[var(--muted)]">当前问题</div>
          <p className="m-0 mt-1 break-words text-[15px] font-medium">{message.question}</p>

          <div className="mt-4">
            <TraceList trace={message.trace} />
          </div>

          <dl className="m-0 mt-4 grid grid-cols-2 gap-2 text-[13px]">
            <div className="rounded-[6px] bg-[var(--surface-muted)] px-3 py-2">
              <dt className="text-[var(--muted)]">回答证据</dt>
              <dd className="m-0 mt-0.5 text-[16px] font-semibold tabular-nums">
                {supportingCount}
              </dd>
            </div>
            <div className="rounded-[6px] bg-[var(--surface-muted)] px-3 py-2">
              <dt className="text-[var(--muted)]">相关资料</dt>
              <dd className="m-0 mt-0.5 text-[16px] font-semibold tabular-nums">
                {relatedCount}
              </dd>
            </div>
            <div className="rounded-[6px] bg-[var(--surface-muted)] px-3 py-2">
              <dt className="text-[var(--muted)]">可回答子问题</dt>
              <dd className="m-0 mt-0.5 text-[16px] font-semibold tabular-nums">
                {answerableCount}
              </dd>
            </div>
            <div className="rounded-[6px] bg-[var(--surface-muted)] px-3 py-2">
              <dt className="text-[var(--muted)]">候选证据</dt>
              <dd className="m-0 mt-0.5 text-[16px] font-semibold tabular-nums">
                {message.evidence.length}
              </dd>
            </div>
          </dl>

          {message.errorMessage && (
            <p className="m-0 mt-3 break-words text-[13px] text-[var(--danger)]">
              {message.errorMessage}
            </p>
          )}
        </div>
      )}
    </aside>
  )
}

function DiscoveryPanel({
  documents,
  discoveries,
  topics,
  canAnalyze,
  analysisBusy,
  analysisError,
  onRegenerate,
  onAsk,
  questionDisabled,
}: {
  documents: DocumentSummary[]
  discoveries: DiscoverySummary[]
  topics: ScopedDiscoveryTopic[]
  canAnalyze: boolean
  analysisBusy: boolean
  analysisError?: string
  onRegenerate: () => void
  onAsk: (question: string) => void
  questionDisabled: boolean
}) {
  const statuses = new Set(discoveries.map((item) => item.status))
  const hasDocuments = documents.length > 0
  const pending = statuses.has('pending') || statuses.has('processing')
  const ready = statuses.has('completed')
  const failed = statuses.has('failed') || statuses.has('cancelled')
  const topicCount = topics.length
  const questionCount = topics.reduce((sum, topic) => sum + topic.questions.length, 0)
  const chunkCount = documents.reduce((sum, document) => sum + document.chunk_count, 0)
  const sortedTopics = [...topics].sort((left, right) => right.chunk_count - left.chunk_count)
  const strongTopics = sortedTopics.slice(0, 3)
  const strongIds = new Set(strongTopics.map((topic) => topic.id))
  const sparseTopics = sortedTopics
    .filter((topic) => topic.chunk_count <= 2 && !strongIds.has(topic.id))
    .slice(0, 3)

  const state = !hasDocuments
    ? { label: '暂无文档', tone: 'neutral' as const }
    : pending
      ? { label: '分析中', tone: 'blue' as const }
      : failed
        ? { label: '分析失败', tone: 'danger' as const }
        : ready
          ? { label: '已生成', tone: 'ok' as const }
          : { label: '未分析', tone: 'neutral' as const }

  return (
    <section className="mt-5 rounded-[8px] border border-[var(--line)] bg-[var(--surface-muted)] p-4" aria-label="知识地图">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h3 className="m-0 text-[16px] font-semibold">知识地图</h3>
            <Badge tone={state.tone} busy={pending}>
              {state.label}
            </Badge>
          </div>
          <p className="m-0 mt-1 text-[14px] text-[var(--muted)]">
            {hasDocuments
              ? `${documents.length} 份文档 · ${chunkCount} 个文本块 · 已识别 ${topicCount} 个主题、${questionCount} 个可问问题`
              : '上传文档后生成基于当前资料的可问清单。'}
          </p>
        </div>
        {hasDocuments && (
          <button
            type="button"
            className="btn"
            onClick={onRegenerate}
            disabled={!canAnalyze || analysisBusy}
          >
            <Icon name="refresh" size={16} />
            {analysisBusy ? '分析中' : ready ? '重新分析' : '生成可问清单'}
          </button>
        )}
      </div>

      {analysisError && (
        <p className="m-0 mt-3 break-words text-[13px] text-[var(--danger)]">{analysisError}</p>
      )}
      {!analysisError && failed && (
        <p className="m-0 mt-3 text-[13px] text-[var(--danger)]">
          上次知识分析未完成，可重新分析。
        </p>
      )}
      {ready && pending && (
        <p className="m-0 mt-3 text-[13px] text-[var(--muted)]">
          部分知识库仍在生成可问清单。
        </p>
      )}

      {topics.length > 0 && (
        <>
          <div className="mt-4 grid grid-cols-2 gap-2 text-[13px] lg:grid-cols-4">
            {[
              { label: '文档', value: documents.length, unit: '份' },
              { label: '文本块', value: chunkCount, unit: '个' },
              { label: '主题', value: topicCount, unit: '个' },
              { label: '可问问题', value: questionCount, unit: '条' },
            ].map((item) => (
              <div key={item.label} className="rounded-[6px] bg-white px-3 py-2">
                <span className="text-[var(--muted)]">{item.label}</span>
                <div className="mt-0.5 text-[16px] font-semibold tabular-nums">
                  {item.value}
                  <span className="ml-1 text-[13px] font-normal text-[var(--muted)]">
                    {item.unit}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {(strongTopics.length > 0 || sparseTopics.length > 0) && (
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {strongTopics.length > 0 && (
                <div className="rounded-[6px] bg-white px-3 py-2.5">
                  <div className="text-[13px] font-medium text-[var(--muted)]">资料较充分</div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {strongTopics.map((topic) => (
                      <span
                        key={`strong-${topic.id}`}
                        className="rounded-[4px] bg-[var(--ok-soft)] px-2 py-0.5 text-[13px] text-[var(--ok)]"
                      >
                        {topic.title}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              {sparseTopics.length > 0 && (
                <div className="rounded-[6px] bg-white px-3 py-2.5">
                  <div className="text-[13px] font-medium text-[var(--muted)]">资料较少</div>
                  <div className="mt-1.5 flex flex-wrap gap-1.5">
                    {sparseTopics.map((topic) => (
                      <span
                        key={`sparse-${topic.id}`}
                        className="rounded-[4px] bg-[var(--warn-soft)] px-2 py-0.5 text-[13px] text-[var(--warn)]"
                      >
                        {topic.title}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="mt-4 grid gap-3">
            {topics.map((topic) => (
              <article key={topic.id} className="rounded-[8px] border border-[var(--line)] bg-white p-3.5">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h4 className="m-0 break-words text-[15px] font-semibold">{topic.title}</h4>
                      <Badge tone="neutral">{topic.type}</Badge>
                      <Badge tone="blue">{topic.chunk_count} 个片段</Badge>
                    </div>
                    {topic.summary && (
                      <p className="m-0 mt-1 break-words text-[13px] text-[var(--muted)]">
                        {topic.summary}
                      </p>
                    )}
                  </div>
                  <span className="text-[13px] text-[var(--muted)] tabular-nums">
                    信心 {Math.round(topic.confidence * 100)}%
                  </span>
                </div>
                {topic.knowledgeBaseName && (
                  <div className="mt-1 text-[13px] text-[var(--muted)]">
                    来源：{topic.knowledgeBaseName}
                  </div>
                )}
                <ul className="m-0 mt-3 grid list-none gap-2 p-0">
                  {topic.questions.map((question) => (
                    <li key={question.id}>
                      <button
                        type="button"
                        className="flex w-full items-center justify-between gap-3 rounded-[6px] border border-[var(--line)] bg-white px-3 py-2 text-left text-[14px] hover:border-[var(--line-hover)] hover:bg-[#f7f9fc] disabled:cursor-not-allowed disabled:opacity-60"
                        onClick={() => onAsk(question.question)}
                        disabled={questionDisabled}
                      >
                        <span className="min-w-0 break-words">{question.question}</span>
                        <span className="flex shrink-0 items-center gap-2 text-[13px] text-[var(--muted)]">
                          {question.source_chunk_ids.length} 个来源
                          <Icon name="send" size={15} />
                        </span>
                      </button>
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </>
      )}

      {hasDocuments && !ready && !pending && !failed && (
        <p className="m-0 mt-3 text-[14px] text-[var(--muted)]">
          尚未生成可问清单。
        </p>
      )}
      {hasDocuments && ready && topics.length === 0 && (
        <p className="m-0 mt-3 text-[14px] text-[var(--muted)]">
          本次资料中没有识别出可用主题。
        </p>
      )}
    </section>
  )
}

export default function Chat({
  bases,
  documents,
  discoveries,
  active,
  disabled,
  goKnowledge,
  onAnalyze,
  analysisError,
}: {
  bases: KnowledgeBase[]
  documents: DocumentSummary[]
  discoveries: Record<string, DiscoverySummary>
  active: boolean
  disabled: boolean
  goKnowledge: () => void
  onAnalyze: (kbId: string) => Promise<boolean>
  analysisError?: string
}) {
  const [selected, setSelected] = useState<string[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [analysisRequesting, setAnalysisRequesting] = useState(false)
  const sessionRef = useRef(0)
  const sequenceRef = useRef(0)
  const controllerRef = useRef<AbortController | null>(null)
  const selectedInitializedRef = useRef(false)
  const endRef = useRef<HTMLDivElement>(null)

  const scope = selected.filter((id) => bases.some((base) => base.id === id))
  const busy = messages.some((message) => message.role === 'assistant' && !message.done)
  const latestAssistant = [...messages].reverse().find((item) => item.role === 'assistant')
  const canAsk = !disabled && scope.length > 0
  const scopeDocuments = useMemo(
    () => documents.filter((document) => scope.includes(document.kb_id)),
    [documents, scope],
  )
  const scopedDiscoveries = useMemo(
    () => scope.flatMap((id) => (discoveries[id] ? [discoveries[id]] : [])),
    [discoveries, scope],
  )
  const discoveryTopics = useMemo(
    () =>
      scopedDiscoveries.flatMap((summary) => {
        const knowledgeBaseName =
          bases.find((base) => base.id === summary.kb_id)?.name ?? ''
        return summary.topics.map((topic) => ({ ...topic, knowledgeBaseName }))
      }),
    [bases, scopedDiscoveries],
  )
  const discoveryPending = scopedDiscoveries.some(
    (summary) => summary.status === 'pending' || summary.status === 'processing',
  )
  const analysisBusy = discoveryPending || analysisRequesting
  const analysisTargetIds = scope.filter((id) =>
    documents.some((document) => document.kb_id === id),
  )
  const canAnalyze = !disabled && analysisTargetIds.length > 0 && !analysisBusy

  const regenerateDiscovery = async () => {
    if (!canAnalyze) return
    setAnalysisRequesting(true)
    await Promise.all(analysisTargetIds.map((id) => onAnalyze(id)))
    setAnalysisRequesting(false)
  }

  useEffect(() => {
    if (!selectedInitializedRef.current && bases.length > 0) {
      selectedInitializedRef.current = true
      setSelected(bases.map((base) => base.id))
    }
  }, [bases])

  useEffect(() => {
    return () => controllerRef.current?.abort()
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: 'end', behavior: 'smooth' })
  }, [messages, busy])

  const patchMessage = (id: number, patch: Partial<AssistantMessage>) => {
    setMessages((list) =>
      list.map((message) =>
        message.id === id && message.role === 'assistant' ? { ...message, ...patch } : message,
      ),
    )
  }

  const applyTraceEvent = (trace: TraceItem[], event: SseEvent): TraceItem[] => {
    if (!event.stage) return trace
    return trace.map((item) => {
      if (item.stage !== event.stage) return item
      if (event.type === 'stage_started') {
        return { ...item, status: 'running', message: event.message }
      }
      if (event.type === 'stage_completed') {
        return { ...item, status: 'completed', message: event.message }
      }
      return item
    })
  }

  const run = async (message: AssistantMessage) => {
    const session = sessionRef.current
    const controller = new AbortController()
    controllerRef.current = controller

    let sawCompleted = false
    let sawError = false
    let sawFinal = false
    let trace = initialTrace()
    let evidence: Evidence[] = []
    let decisions: EvidenceDecision[] = []
    let final: FinalAnswer | undefined
    let errorMessage: string | undefined

    patchMessage(message.id, {
      running: true,
      done: false,
      trace: initialTrace(),
      evidence: [],
      decisions: [],
      final: undefined,
      errorMessage: undefined,
      traceOpen: false,
      evidenceOpen: false,
    })

    const handleEvent = (event: SseEvent) => {
      if (session !== sessionRef.current) return
      trace = applyTraceEvent(trace, event)

      if (event.type === 'evidence' && event.payload.evidence) {
        evidence = event.payload.evidence
      } else if (event.type === 'decision' && event.payload.decisions) {
        decisions = event.payload.decisions
      } else if (event.type === 'answer_final') {
        final = event.payload as FinalAnswer
        sawFinal = true
        trace = trace.map((item) =>
          item.status === 'pending' && item.stage !== 'regenerate'
            ? { ...item, status: 'skipped', message: '未触发' }
            : item,
        )
      } else if (event.type === 'completed') {
        sawCompleted = true
      } else if (event.type === 'error') {
        sawError = true
        errorMessage = event.message || '本次处理失败，请稍后重试。'
        trace = trace.map((item) => {
          if (item.status === 'running') return { ...item, status: 'failed', message: event.message }
          if (item.status === 'pending') return { ...item, status: 'skipped', message: '因错误终止' }
          return item
        })
      }

      patchMessage(message.id, {
        trace,
        evidence,
        decisions,
        final,
        errorMessage,
        running: !(sawCompleted || sawError),
        done: sawCompleted || sawError,
      })
    }

    try {
      await streamChat(
        {
          question: message.question,
          history: message.historySnapshot,
          kb_ids: message.kbIds,
        },
        handleEvent,
        controller.signal,
      )
    } catch (error) {
      if (session === sessionRef.current) {
        const safeError =
          error instanceof ApiError ? error : new ApiError('network_error', '本次处理失败，请稍后重试。')
        patchMessage(message.id, {
          running: false,
          done: true,
          errorMessage: safeError.message,
        })
      }
      return
    }

    if (session !== sessionRef.current) return
    if (!sawCompleted && !sawError) {
      patchMessage(message.id, {
        running: false,
        done: true,
        errorMessage: sawFinal
          ? undefined
          : '连接中断，本次处理未完成。',
      })
    }
  }

  const ask = (text: string) => {
    const question = text.trim()
    if (!question || !canAsk || busy) return

    const history: ChatHistoryMessage[] = []
    for (const message of messages.slice(-10)) {
      if (message.role === 'user') {
        history.push({ role: 'user', content: message.text })
      } else if (message.final) {
        history.push({ role: 'assistant', content: message.final.answer })
      }
    }

    const userId = ++sequenceRef.current
    const assistantId = ++sequenceRef.current
    const assistant: AssistantMessage = {
      id: assistantId,
      role: 'assistant',
      question,
      kbIds: scope,
      historySnapshot: history,
      running: true,
      done: false,
      trace: initialTrace(),
      evidence: [],
      decisions: [],
      evidenceOpen: false,
      traceOpen: false,
    }
    setMessages((list) => [...list, { id: userId, role: 'user', text: question }, assistant])
    setInput('')
    void run(assistant)
  }

  const retry = (message: Extract<ChatMessage, { role: 'assistant' }>) => {
    if (busy || disabled) return
    void run(message)
  }

  const reset = () => {
    sessionRef.current += 1
    controllerRef.current?.abort()
    controllerRef.current = null
    setMessages([])
    setInput('')
  }

  const toggle = (id: string) => {
    setSelected((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    )
  }

  return (
    <div hidden={!active}>
      <PageHeading title="问答" sub="在选定知识库中检索，并查看回答依据。">
        <button className="btn" type="button" onClick={reset} disabled={!messages.length}>
          <Icon name="refresh" size={16} />
          新会话
        </button>
      </PageHeading>

      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_360px]">
        <section className="card flex min-h-[560px] flex-col overflow-hidden" aria-label="会话">
          <div className="border-b border-[var(--line)] px-4 py-4 sm:px-5">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <span className="text-[14px] font-medium" id="scope-label">
                检索范围
              </span>
              {bases.length === 0 ? (
                <span className="text-[14px] text-[var(--muted)]">
                  暂无知识库。
                  <button
                    type="button"
                    className="cursor-pointer border-0 bg-transparent p-0 text-[var(--primary)] underline"
                    onClick={goKnowledge}
                  >
                    前往创建
                  </button>
                </span>
              ) : (
                <div role="group" aria-labelledby="scope-label" className="flex flex-wrap gap-2">
                  {bases.map((base) => {
                    const on = scope.includes(base.id)
                    return (
                      <button
                        key={base.id}
                        type="button"
                        role="checkbox"
                        aria-checked={on}
                        onClick={() => toggle(base.id)}
                        className={`inline-flex min-h-8 items-center gap-2 rounded-full border px-3 text-left text-[14px] ${
                          on
                            ? 'border-[var(--primary)] bg-[var(--primary-soft)] font-medium text-[var(--primary)]'
                            : 'border-[var(--line-strong)] bg-white text-[var(--ink)] hover:bg-[#f3f5f8]'
                        }`}
                      >
                        <span
                          className={`grid h-4 w-4 place-items-center rounded-[3px] border ${
                            on
                              ? 'border-[var(--primary)] bg-[var(--primary)] text-white'
                              : 'border-[#9aa3b0]'
                          }`}
                        >
                          {on && <Icon name="check" size={11} />}
                        </span>
                        <span className="break-all">{base.name}</span>
                        <span className="text-[13px] font-normal text-[var(--muted)]">
                          {base.document_count} 份文档
                        </span>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
            {disabled && bases.length > 0 && (
              <p className="m-0 mt-2 text-[13px] text-[var(--warn)]">
                服务配置或数据状态异常，暂时不能提问。
              </p>
            )}
            {!disabled && scope.length === 0 && bases.length > 0 && (
              <p className="m-0 mt-2 text-[13px] text-[var(--warn)]">请先选择至少一个知识库</p>
            )}
          </div>

          <div className="flex-1 px-4 py-5 sm:px-5">
            {messages.length === 0 ? (
              <div className="mx-auto max-w-[720px] py-8">
                <h2 className="t-section m-0">还没有提问</h2>
                <p className="m-0 mt-1 text-[15px] text-[var(--muted)]">
                  回答只依据所选知识库中的文档，证据不足时会明确拒答。
                </p>
                <DiscoveryPanel
                  documents={scopeDocuments}
                  discoveries={scopedDiscoveries}
                  topics={discoveryTopics}
                  canAnalyze={canAnalyze}
                  analysisBusy={analysisBusy}
                  analysisError={analysisError}
                  onRegenerate={() => void regenerateDiscovery()}
                  onAsk={ask}
                  questionDisabled={!canAsk || busy}
                />
              </div>
            ) : (
              <div className="grid gap-6">
                {messages.map((message) => {
                  if (message.role === 'user') {
                    return (
                      <div key={message.id} className="flex justify-end">
                        <div className="max-w-[85%] break-words rounded-[8px] bg-[var(--neutral-soft)] px-4 py-2.5 text-[15px]">
                          {message.text}
                        </div>
                      </div>
                    )
                  }

                  const final = message.final
                  const finalParts = Array.isArray(final?.parts) ? final.parts : []
                  const verifiedParts = finalParts.filter((part) => part.status === 'verified')
                  const refusedParts = finalParts.filter((part) => part.status === 'refused')
                  const failed = Boolean(message.errorMessage)
                  const answerState = failed
                    ? { label: '处理失败', tone: 'danger' as const }
                    : !final
                      ? { label: '处理中', tone: 'neutral' as const }
                      : final.refused
                        ? { label: '全部拒答', tone: 'neutral' as const }
                        : refusedParts.length > 0
                          ? { label: '局部拒答', tone: 'warn' as const }
                          : { label: '回答完成', tone: 'ok' as const }

                  return (
                    <article key={message.id} className="min-w-0" aria-live="polite">
                      {!message.done ? (
                        <div className="rounded-[8px] border border-[var(--line)] bg-white p-4">
                          <div className="mb-3 flex flex-wrap items-center gap-3">
                            <Badge tone="neutral" busy>
                              处理中
                            </Badge>
                            <span className="text-[13px] text-[var(--muted)]">
                              校验完成后再显示回答
                            </span>
                          </div>
                          <TraceList trace={message.trace} />
                        </div>
                      ) : (
                        <div>
                          <div className="rounded-[8px] border border-[var(--line)] bg-white p-4">
                            <Badge tone={answerState.tone}>{answerState.label}</Badge>
                            {failed && (
                              <div className="mt-3 flex items-start gap-2 text-[15px]">
                                <span className="mt-1 text-[var(--danger)]">
                                  <Icon name="alert" size={16} />
                                </span>
                                <p className="m-0 break-words">{message.errorMessage}</p>
                              </div>
                            )}
                            {final?.refused && (
                              <p className="m-0 mt-3 text-[15px]">
                                所选知识库中没有足够证据回答这个问题，因此没有生成回答。
                              </p>
                            )}
                            {verifiedParts.map((part) => (
                              <div key={part.subquestion_id} className="mt-3 first:mt-0">
                                <div className="text-[13px] font-medium text-[var(--muted)]">
                                  {part.question}
                                </div>
                                <p className="m-0 whitespace-pre-wrap break-words text-[15px] leading-7">
                                  {part.answer}
                                </p>
                              </div>
                            ))}
                            {!failed && refusedParts.length > 0 && (
                              <div className="mt-3 rounded-[6px] border-l-2 border-[#d6a74f] bg-[var(--warn-soft)] px-3 py-2">
                                <div className="text-[14px] font-medium text-[var(--warn)]">
                                  {final?.refused
                                    ? '以下问题缺少充分证据'
                                    : '以下部分缺少充分证据，未作回答'}
                                </div>
                                <ul className="m-0 mt-1 list-disc break-words pl-5 text-[14px]">
                                  {refusedParts.map((part) => (
                                    <li key={part.subquestion_id}>
                                      {part.question}：{part.refusal_reason}
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}
                            {final?.truncated && (
                              <p className="m-0 mt-3 text-[13px] text-[var(--muted)]">
                                子问题较多，本次仅处理前 5 项。
                              </p>
                            )}
                          </div>
                          <div className="mt-1.5 flex flex-wrap items-center gap-1">
                            {!failed && (
                              <button
                                type="button"
                                className="btn btn-quiet"
                                aria-pressed={message.evidenceOpen}
                                onClick={() =>
                                  patchMessage(message.id, { evidenceOpen: !message.evidenceOpen })
                                }
                              >
                                查看证据（{message.evidence.length}）
                                <span className={message.evidenceOpen ? 'rotate-180' : ''}>
                                  <Icon name="down" size={14} />
                                </span>
                              </button>
                            )}
                            <button
                              type="button"
                              className="btn btn-quiet"
                              aria-pressed={message.traceOpen}
                              onClick={() =>
                                patchMessage(message.id, { traceOpen: !message.traceOpen })
                              }
                            >
                              查看执行过程
                              <span className={message.traceOpen ? 'rotate-180' : ''}>
                                <Icon name="down" size={14} />
                              </span>
                            </button>
                            {failed && (
                              <button
                                type="button"
                                className="btn btn-quiet"
                                onClick={() => retry(message)}
                              >
                                <Icon name="refresh" size={14} />
                                重试
                              </button>
                            )}
                          </div>
                          {message.evidenceOpen && (
                            <div className="mt-2 rounded-[8px] border border-[var(--line)] bg-[var(--surface-muted)] p-3.5">
                              <EvidenceList evidence={message.evidence} />
                            </div>
                          )}
                          {message.traceOpen && (
                            <div className="mt-2 rounded-[8px] border border-[var(--line)] bg-[var(--surface-muted)] p-4">
                              <TraceList trace={message.trace} />
                            </div>
                          )}
                        </div>
                      )}
                    </article>
                  )
                })}
                <div ref={endRef} />
              </div>
            )}
          </div>

          <form
            className="border-t border-[var(--line)] bg-[var(--surface-muted)] p-4"
            onSubmit={(event) => {
              event.preventDefault()
              ask(input)
            }}
          >
            <div
              className={`w-full rounded-[8px] border bg-white p-3 focus-within:border-[var(--primary)] ${
                canAsk ? 'border-[var(--line-strong)]' : 'border-[var(--line)] bg-[#f3f4f6]'
              }`}
            >
              <textarea
                aria-label="输入问题"
                className="block h-[56px] w-full resize-none border-0 bg-transparent text-[15px] outline-none disabled:cursor-not-allowed"
                placeholder="输入问题"
                value={input}
                disabled={!canAsk}
                maxLength={4000}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault()
                    ask(input)
                  }
                }}
              />
              <div className="mt-1 flex flex-wrap items-center justify-between gap-3">
                <span className="text-[13px] text-[var(--muted)]">
                  {disabled
                    ? '服务状态异常，暂不能提问'
                    : canAsk
                      ? 'Enter 提交 · Shift + Enter 换行'
                      : '请先选择至少一个知识库'}
                </span>
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={!canAsk || !input.trim() || busy}
                >
                  提问
                </button>
              </div>
            </div>
          </form>
        </section>

        <ExecutionPanel message={latestAssistant} />
      </div>
    </div>
  )
}
