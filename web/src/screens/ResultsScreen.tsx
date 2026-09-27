import { useQuery } from '@tanstack/react-query'
import { useState } from 'react'
import { api, blobUrl, thumbUrl, type Result } from '../api'

function expiresIn(result: Result): string | null {
  if (!result.expires_at) return null
  const hours = (new Date(result.expires_at).getTime() - Date.now()) / 3_600_000
  if (hours <= 0) return 'expiring'
  return hours >= 1
    ? `${String(Math.floor(hours))}h left`
    : `${String(Math.ceil(hours * 60))}m left`
}

export function ResultsScreen() {
  const results = useQuery({ queryKey: ['results'], queryFn: () => api.results() })
  const [open, setOpen] = useState<Result | null>(null)

  if (results.isPending) return <p className="muted">Loading…</p>
  if (results.error) return <p role="alert">{results.error.message}</p>
  if (results.data.results.length === 0) return <p className="muted">No results yet.</p>

  return (
    <>
      <div className="grid">
        {results.data.results.map((r) => (
          <button
            key={r.id}
            type="button"
            className="tile"
            onClick={() => {
              setOpen(r)
            }}
          >
            <img
              src={thumbUrl(r.blob_sha)}
              alt={String(r.spec?.params.prompt ?? '')}
              loading="lazy"
            />
            {expiresIn(r) && <span className="expiry">{expiresIn(r)}</span>}
          </button>
        ))}
      </div>
      {open && (
        <ResultDetail
          result={open}
          onClose={() => {
            setOpen(null)
          }}
        />
      )}
    </>
  )
}

function ResultDetail({ result, onClose }: { result: Result; onClose: () => void }) {
  const params = result.spec?.params ?? {}
  const share = async () => {
    const blob = await (await fetch(blobUrl(result.blob_sha))).blob()
    const file = new File([blob], `degas-${String(result.seed)}.png`, { type: result.media_type })
    if ('canShare' in navigator && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file] })
    } else {
      const a = document.createElement('a')
      a.href = blobUrl(result.blob_sha)
      a.download = file.name
      a.click()
    }
  }
  return (
    <div className="sheet" role="dialog" aria-label="Result">
      <div className="sheet-bar">
        <button type="button" className="secondary" onClick={onClose}>
          Close
        </button>
        <button
          type="button"
          onClick={() => {
            void share().catch(() => undefined)
          }}
        >
          Save to Photos
        </button>
      </div>
      <img className="full" src={blobUrl(result.blob_sha)} alt={String(params.prompt ?? '')} />
      <dl className="config">
        <dt>Prompt</dt>
        <dd>{String(params.prompt ?? '')}</dd>
        {params.negative_prompt ? (
          <>
            <dt>Negative</dt>
            <dd>{String(params.negative_prompt)}</dd>
          </>
        ) : null}
        <dt>Model</dt>
        <dd>{result.spec?.model.path.split('/').pop()}</dd>
        <dt>Seed</dt>
        <dd>{result.seed}</dd>
        <dt>Size</dt>
        <dd>
          {result.width}×{result.height}
        </dd>
        <dt>Steps / CFG</dt>
        <dd>
          {String(params.steps)} / {String(params.cfg)} · {String(params.scheduler)}
        </dd>
      </dl>
    </div>
  )
}
