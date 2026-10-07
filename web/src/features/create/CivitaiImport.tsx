import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'

import { api } from '@/api/client'
import { queries } from '@/api/queries'
import type { Asset, CivitaiPlan } from '@/api/types'
import { formatBytes } from '@/lib/format'

export interface CivitaiImportProps {
  /** The family Create is on: an import for another one says where it went. */
  family: string
  /**
   * An import started here finished and the index has it. Returns whether it was added to
   * the form (it isn't when it's for another family or variant).
   */
  onImported: (paths: string[], index: Asset[]) => boolean
}

/**
 * Import a LoRA from Civitai or Hugging Face, at the foot of the LoRA picker: paste a link,
 * check what it is, then copy it into Drive. The copy runs on the server, so closing the sheet doesn't stop it.
 */
export function CivitaiImport({ family, onImported }: CivitaiImportProps) {
  const qc = useQueryClient()
  const [open, setOpen] = useState(false)
  const [url, setUrl] = useState('')
  const [pasteFailed, setPasteFailed] = useState(false)
  // The import started from this sheet, and what became of it once it finished.
  const [started, setStarted] = useState<string | null>(null)
  const [added, setAdded] = useState<boolean | null>(null)
  const handled = useRef<string | null>(null)
  const families = useQuery(queries.families())
  const job = useQuery(queries.civitaiImport()).data
  const check = useMutation({ mutationFn: (link: string) => api.civitaiPlan(link, family) })
  const start = useMutation({
    mutationFn: (link: string) => api.civitaiImport(link, family),
    onSuccess: (j) => {
      qc.setQueryData(['civitai-import'], j)
      setStarted(j.id)
      setAdded(null)
      setUrl('')
      check.reset()
    },
  })
  const canPaste = typeof navigator !== 'undefined' && 'clipboard' in navigator
  const familyLabel = (id: string) => families.data?.find((f) => f.id === id)?.label ?? id

  useEffect(() => {
    if (job?.id !== started || job.state !== 'done' || handled.current === job.id) return
    handled.current = job.id
    void qc.query({ ...queries.assets(), staleTime: 0 }).then((index) => {
      const indexed = index.some((a) => job.paths.includes(a.path))
      setAdded(indexed ? onImported(job.paths, index) : null)
    })
  }, [job, started, qc, onImported])

  const running = job?.state === 'copying' || job?.state === 'finishing'
  if (running) {
    const pct = job.total ? Math.floor((100 * job.done) / job.total) : 0
    return (
      <section className="civitai" aria-label="Import a LoRA">
        <p className="asset-name">{job.label}</p>
        <progress max={job.total || 1} value={job.state === 'finishing' ? job.total : job.done} />
        <p className="asset-meta" aria-live="polite">
          {job.state === 'finishing'
            ? 'Adding its preview and rescanning Drive…'
            : `Copying to Drive, ${String(pct)}% of ${formatBytes(job.total)}`}
        </p>
      </section>
    )
  }

  const plan = check.data
  const finished = job?.id === started ? job : null
  return (
    <section className="civitai" aria-label="Import a LoRA">
      {finished?.state === 'done' && (
        <p className="civitai-done" role="status">
          Imported {finished.label}
          {added ? ' and added it.' : '.'}
          {added === false &&
            (finished.family === family
              ? ' This model can’t use it.'
              : ` It’s in ${familyLabel(finished.family)}’s LoRAs.`)}
          {finished.warnings.map((w) => ` ${w}.`).join('')}
        </p>
      )}
      {finished?.state === 'failed' && <p role="alert">{finished.error}</p>}
      {!open ? (
        <button
          type="button"
          className="btn quiet small"
          onClick={() => {
            setOpen(true)
          }}
        >
          Import a LoRA
        </button>
      ) : plan ? (
        <PlanSummary
          plan={plan}
          elsewhere={plan.family === family ? null : familyLabel(plan.family)}
          pending={start.isPending}
          onImport={() => {
            start.mutate(url.trim())
          }}
          onCancel={() => {
            check.reset()
          }}
        />
      ) : (
        // Not a <form>: the sheet renders inside Create's.
        <div className="civitai-link">
          <div className="link-row">
            <input
              type="url"
              inputMode="url"
              aria-label="Civitai or Hugging Face link"
              placeholder="civitai.com/models/… or huggingface.co/…"
              autoCapitalize="none"
              autoCorrect="off"
              value={url}
              onChange={(e) => {
                setUrl(e.target.value)
              }}
              onKeyDown={(e) => {
                if (e.key !== 'Enter') return
                e.preventDefault()
                if (url.trim()) check.mutate(url.trim())
              }}
            />
            {canPaste && (
              <button
                type="button"
                className="btn quiet"
                onClick={() => {
                  setPasteFailed(false)
                  navigator.clipboard
                    .readText()
                    .then((text) => {
                      setUrl(text.trim())
                    })
                    .catch(() => {
                      setPasteFailed(true)
                    })
                }}
              >
                Paste
              </button>
            )}
          </div>
          <button
            type="button"
            className="btn"
            disabled={!url.trim() || check.isPending}
            onClick={() => {
              check.mutate(url.trim())
            }}
          >
            {check.isPending ? 'Checking…' : 'Check link'}
          </button>
          {pasteFailed && <p role="alert">Couldn’t read the clipboard. Paste into the field.</p>}
        </div>
      )}
      {check.error && <p role="alert">{check.error.message}</p>}
      {start.error && <p role="alert">{start.error.message}</p>}
    </section>
  )
}

function PlanSummary({
  plan,
  elsewhere,
  pending,
  onImport,
  onCancel,
}: {
  plan: CivitaiPlan
  elsewhere: string | null
  pending: boolean
  onImport: () => void
  onCancel: () => void
}) {
  const size = plan.files.reduce((n, f) => n + f.size, 0)
  const halves = plan.files.flatMap((f) => (f.half ? [f.half] : []))
  const folder = `degas/loras/${plan.family}/`
  return (
    <div className="civitai-plan">
      <p className="asset-name">{plan.label}</p>
      <p className="asset-meta">
        {[
          plan.version_name,
          plan.base_model,
          halves.length === 2 ? 'high- and low-noise pair' : halves[0] && `${halves[0]}-noise half`,
          formatBytes(size),
        ]
          .filter(Boolean)
          .join(', ')}
      </p>
      {plan.trigger_words.length > 0 && (
        <p className="asset-meta">Trigger words: {plan.trigger_words.join(', ')}</p>
      )}
      <p className="asset-meta">
        Goes in <code>{folder}</code>
        {elsewhere && `, for ${elsewhere}, so it won’t be listed for this model`}.
      </p>
      {plan.warnings.map((w) => (
        <p key={w} className="civitai-warn">
          {w}.
        </p>
      ))}
      <div className="civitai-actions">
        <button type="button" className="btn quiet" onClick={onCancel}>
          Cancel
        </button>
        <button type="button" className="btn" disabled={pending} onClick={onImport}>
          {pending ? 'Starting…' : 'Import'}
        </button>
      </div>
    </div>
  )
}
