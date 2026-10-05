import { useEffect, type ReactNode } from 'react'

const paths: Record<string, ReactNode> = {
  grid: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1" />
      <rect x="14" y="3" width="7" height="7" rx="1" />
      <rect x="3" y="14" width="7" height="7" rx="1" />
      <rect x="14" y="14" width="7" height="7" rx="1" />
    </>
  ),
  book: (
    <path d="M4 4h6a2 2 0 0 1 2 2v14a2 2 0 0 0-2-2H4V4Zm16 0h-6a2 2 0 0 0-2 2v14a2 2 0 0 1 2-2h6V4Z" />
  ),
  chat: <path d="M20 12a7 7 0 0 1-7 7H8l-4 2V7a3 3 0 0 1 3-3h10a3 3 0 0 1 3 3v5Z" />,
  file: <path d="M14 3H6v18h12V7l-4-4Zm0 0v4h4" />,
  layers: <path d="m12 3 9 5-9 5-9-5 9-5ZM3 13l9 5 9-5" />,
  plus: <path d="M12 5v14M5 12h14" />,
  upload: <path d="M12 16V4m-5 5 5-5 5 5M5 20h14" />,
  trash: <path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" />,
  close: <path d="m6 6 12 12M18 6 6 18" />,
  check: <path d="m5 12 4 4 10-10" />,
  send: <path d="M5 12h14m-6-6 6 6-6 6" />,
  alert: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v6m0 3h.01" />
    </>
  ),
  down: <path d="m6 9 6 6 6-6" />,
  refresh: <path d="M20 11a8 8 0 1 0-2 5.3M20 4v7h-7" />,
}

export function Icon({ name, size = 18 }: { name: string; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      className="shrink-0"
    >
      {paths[name] ?? paths.file}
    </svg>
  )
}

export function Badge({
  tone = 'neutral',
  busy,
  children,
}: {
  tone?: 'neutral' | 'ok' | 'warn' | 'danger' | 'blue'
  busy?: boolean
  children: ReactNode
}) {
  return (
    <span className={`badge ${tone === 'neutral' ? '' : tone} ${busy ? 'busy' : ''}`}>
      {children}
    </span>
  )
}

export function PageHeading({
  title,
  sub,
  children,
}: {
  title: string
  sub: string
  children?: ReactNode
}) {
  return (
    <div className="mb-6 flex flex-wrap items-start justify-between gap-x-6 gap-y-3">
      <div className="min-w-0">
        <h1 className="t-title m-0">{title}</h1>
        <p className="mt-1 text-[15px] text-[var(--muted)]">{sub}</p>
      </div>
      {children && <div className="flex shrink-0 flex-wrap items-center gap-2">{children}</div>}
    </div>
  )
}

export function Modal({
  title,
  onClose,
  children,
  actions,
}: {
  title: string
  onClose: () => void
  children: ReactNode
  actions: ReactNode
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      className="fixed inset-0 z-30 flex items-center justify-center bg-[#1f232880] p-5"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose()
      }}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="card w-full max-w-[440px] p-6 shadow-xl"
      >
        <h2 className="t-section m-0 mb-3">{title}</h2>
        {children}
        <div className="mt-6 flex justify-end gap-2">{actions}</div>
      </div>
    </div>
  )
}
