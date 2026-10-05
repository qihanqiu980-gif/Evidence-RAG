import type {
  ChatRequest,
  DocumentBatchResult,
  KnowledgeBase,
  SseEvent,
  SystemStatus,
  DocumentSummary,
  UploadJobSummary,
} from './types'

export class ApiError extends Error {
  readonly code: string

  constructor(code: string, message: string) {
    super(message)
    this.code = code
  }
}

const safeMessage = '本地服务暂时不可用，请稍后重试。'

async function parseError(response: Response): Promise<ApiError> {
  try {
    const payload = await response.json()
    const detail = payload?.detail
    if (detail && typeof detail.code === 'string' && typeof detail.message === 'string') {
      return new ApiError(detail.code, detail.message)
    }
  } catch {
    // Use the safe fallback below.
  }
  return new ApiError(`http_${response.status}`, safeMessage)
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData ? {} : { Accept: 'application/json' }),
        ...(init?.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...init?.headers,
      },
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('network_error', safeMessage)
  }
  if (!response.ok) throw await parseError(response)
  if (response.status === 204) return undefined as T
  try {
    return (await response.json()) as T
  } catch {
    throw new ApiError('invalid_response', safeMessage)
  }
}

export const getStatus = () => request<SystemStatus>('/api/status')

export const listKnowledgeBases = () =>
  request<KnowledgeBase[]>('/api/knowledge-bases')

export const createKnowledgeBase = (name: string) =>
  request<KnowledgeBase>('/api/knowledge-bases', {
    method: 'POST',
    body: JSON.stringify({ name }),
  })

export const deleteKnowledgeBase = (id: string) =>
  request<void>(`/api/knowledge-bases/${encodeURIComponent(id)}`, {
    method: 'DELETE',
  })

export const listDocuments = (kbId: string) =>
  request<DocumentSummary[]>(
    `/api/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
  )

async function requestBatch(path: string, init?: RequestInit): Promise<DocumentBatchResult> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch {
    throw new ApiError('network_error', safeMessage)
  }
  if (![200, 201, 207, 409].includes(response.status)) throw await parseError(response)
  try {
    return (await response.json()) as DocumentBatchResult
  } catch {
    throw new ApiError('invalid_response', safeMessage)
  }
}

export const uploadDocuments = (kbId: string, files: File[]) => {
  const body = new FormData()
  files.forEach((file) => body.append('files', file))
  return requestBatch(
    `/api/knowledge-bases/${encodeURIComponent(kbId)}/documents`,
    { method: 'POST', body },
  )
}

export const uploadDocumentsAsync = (kbId: string, files: File[]) => {
  const body = new FormData()
  files.forEach((file) => body.append('files', file))
  return request<{ job_id: string }>(
    `/api/knowledge-bases/${encodeURIComponent(kbId)}/documents/async`,
    { method: 'POST', body },
  )
}

export const getUploadJob = (jobId: string) =>
  request<UploadJobSummary>(`/api/jobs/${encodeURIComponent(jobId)}`)

export const cancelUploadJob = (jobId: string) =>
  request<UploadJobSummary>(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
  })

export const importDemoDocuments = (kbId: string) =>
  requestBatch(
    `/api/knowledge-bases/${encodeURIComponent(kbId)}/import-demo`,
    { method: 'POST' },
  )

export const deleteDocument = (kbId: string, documentId: string) =>
  request<void>(
    `/api/knowledge-bases/${encodeURIComponent(kbId)}/documents/${encodeURIComponent(documentId)}`,
    { method: 'DELETE' },
  )

function parseSseBlock(block: string): SseEvent | null {
  const data = block
    .split(/\r?\n/)
    .filter((line) => line.startsWith('data:'))
    .map((line) => line.slice(5).startsWith(' ') ? line.slice(6) : line.slice(5))
    .join('\n')
  if (!data) return null
  try {
    return JSON.parse(data) as SseEvent
  } catch {
    return null
  }
}

export async function streamChat(
  requestPayload: ChatRequest,
  onEvent: (event: SseEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  let response: Response
  try {
    response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {
        Accept: 'text/event-stream',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestPayload),
      signal,
    })
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error
    throw new ApiError('network_error', safeMessage)
  }
  if (!response.ok) throw await parseError(response)
  if (!response.body) throw new ApiError('stream_unavailable', safeMessage)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const normalized = buffer.replace(/\r\n/g, '\n')
    const blocks = normalized.split('\n\n')
    buffer = blocks.pop() ?? ''
    for (const block of blocks) {
      const event = parseSseBlock(block)
      if (event) onEvent(event)
    }
  }

  buffer += decoder.decode()
  for (const block of buffer.split(/\r?\n\r?\n/)) {
    const event = parseSseBlock(block)
    if (event) onEvent(event)
  }
}
