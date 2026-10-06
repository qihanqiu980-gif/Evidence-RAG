export type BackendStatus = 'ready' | 'configuration_missing' | 'storage_inconsistent'
export type ServiceState = BackendStatus | 'connecting' | 'error'

export type UploadLimits = {
  max_files_per_upload: number
  max_file_bytes: number
  accepted_extensions: string[]
}

export type SystemStatus = {
  status: BackendStatus
  knowledge_base_count: number
  document_count: number
  chunk_count: number
  limits: UploadLimits
}

export type KnowledgeBase = {
  id: string
  name: string
  document_count: number
  chunk_count: number
  created_at: string
}

export type DocumentSummary = {
  id: string
  kb_id: string
  filename: string
  status: 'ready'
  chunk_count: number
  created_at: string
}

export type UploadedDocument = DocumentSummary & {
  progress: string[]
}

export type DocumentUploadError = {
  filename: string
  code: string
  message: string
}

export type DocumentBatchResult = {
  documents: UploadedDocument[]
  errors: DocumentUploadError[]
}

export type UploadJobItemStatus =
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type UploadJobItem = {
  filename: string
  status: UploadJobItemStatus
  code: string | null
  message: string | null
  progress: string[]
  document: DocumentSummary | null
}

export type UploadJobSummary = {
  job_id: string
  kb_id: string
  status: 'pending' | 'processing' | 'completed' | 'cancelled'
  created_at: string
  updated_at: string
  items: UploadJobItem[]
}

export type DiscoveryStatus =
  | 'not_analyzed'
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled'

export type DiscoveryQuestion = {
  id: string
  question: string
  source_chunk_ids: string[]
}

export type DiscoveryTopic = {
  id: string
  title: string
  type: string
  summary: string
  document_ids: string[]
  chunk_count: number
  confidence: number
  updated_at: string
  questions: DiscoveryQuestion[]
}

export type DiscoverySummary = {
  kb_id: string
  status: DiscoveryStatus
  job_id: string | null
  code: string | null
  message: string | null
  document_count: number
  chunk_count: number
  topic_count: number
  question_count: number
  analyzed_at: string | null
  topics: DiscoveryTopic[]
}

export type ChatHistoryMessage = {
  role: 'user' | 'assistant'
  content: string
}

export type ChatRequest = {
  question: string
  history: ChatHistoryMessage[]
  kb_ids: string[]
}

export type WorkflowStage =
  | 'decompose'
  | 'retrieve'
  | 'rerank'
  | 'judge'
  | 'generate'
  | 'validate'
  | 'regenerate'
  | 'compose'

export type Evidence = {
  reference_id: number
  document_id: string
  document_name: string
  heading_path: string[]
  chunk_index: number
  content: string
  similarity: number
  rerank_score: number
  subquestion_id: string
  subquestion: string
  support_status: 'supporting' | 'related'
}

export type EvidenceDecision = {
  subquestion_id: string
  answerable: boolean
  evidence_ids: number[]
  missing_information: string
  reason: string
}

export type FinalAnswerPart = {
  subquestion_id: string
  question: string
  answer: string
  citations: number[]
  status: 'verified' | 'refused'
  refusal_reason: string
}

export type FinalAnswer = {
  answer: string
  refused: boolean
  truncated: boolean
  parts: FinalAnswerPart[]
}

export type SseEvent = {
  type:
    | 'stage_started'
    | 'stage_completed'
    | 'evidence'
    | 'decision'
    | 'answer_final'
    | 'completed'
    | 'error'
  stage: WorkflowStage | null
  message: string
  sequence: number
  payload: {
    evidence?: Evidence[]
    decisions?: EvidenceDecision[]
    answer?: string
    code?: string
    [key: string]: unknown
  }
}
