import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api, isActive } from '../api'

const GiB = 1024 ** 3

function useNow(intervalMs: number) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => {
      setNow(Date.now())
    }, intervalMs)
    return () => {
      clearInterval(id)
    }
  }, [intervalMs])
  return now
}

function countdown(deadline: string | null, now: number): string | null {
  if (!deadline) return null
  const s = Math.max(0, Math.round((new Date(deadline).getTime() - now) / 1000))
  return `${String(Math.floor(s / 60))}:${String(s % 60).padStart(2, '0')}`
}

export function SessionScreen() {
  const qc = useQueryClient()
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  const drive = useQuery({ queryKey: ['drive'], queryFn: api.drive })
  const [gpu, setGpu] = useState('L4')
  const [highMem, setHighMem] = useState(false)
  const now = useNow(1000)

  const onSettled = () => qc.invalidateQueries({ queryKey: ['session'] })
  const start = useMutation({
    mutationFn: () => api.startSession(gpu, highMem),
    onSettled,
  })
  const stop = useMutation({ mutationFn: api.stopSession, onSettled })
  const touch = useMutation({ mutationFn: api.touchSession, onSettled })
  const reset = useMutation({ mutationFn: api.resetWorker, onSettled })
  const rescan = useMutation({
    mutationFn: api.rescan,
    onSettled: () => qc.invalidateQueries({ queryKey: ['drive'] }),
  })

  if (session.isPending) return <p className="muted">Loading…</p>
  if (session.error) return <p role="alert">{session.error.message}</p>

  const snap = session.data
  const s = snap.session
  const active = isActive(snap)
  const remaining = countdown(snap.idle_deadline, now)
  const error = start.error ?? stop.error ?? reset.error

  return (
    <div className="session">
      <section className="card">
        <h2>GPU session</h2>
        {s ? (
          <dl className="config">
            <dt>State</dt>
            <dd>
              <span className={`badge ${s.state}`}>{s.state}</span>
              {snap.step && <span className="muted"> {snap.step}…</span>}
            </dd>
            <dt>GPU</dt>
            <dd>
              {s.gpu}
              {s.high_mem ? ' · high-mem' : ''}
              {snap.worker?.gpu ? ` · ${snap.worker.gpu}` : ''}
            </dd>
            {snap.worker?.vram_total != null && snap.worker.vram_free != null && (
              <>
                <dt>VRAM free</dt>
                <dd>
                  {(snap.worker.vram_free / GiB).toFixed(1)} /{' '}
                  {(snap.worker.vram_total / GiB).toFixed(1)} GiB
                </dd>
              </>
            )}
            {remaining && (
              <>
                <dt>Idle stop in</dt>
                <dd>{remaining}</dd>
              </>
            )}
            {s.error && (
              <>
                <dt>Error</dt>
                <dd className="error">{s.error}</dd>
              </>
            )}
          </dl>
        ) : (
          <p className="muted">No session has been started yet.</p>
        )}

        {active ? (
          <div className="row">
            <button
              type="button"
              className="danger"
              disabled={stop.isPending}
              onClick={() => {
                stop.mutate()
              }}
            >
              {stop.isPending ? 'Stopping…' : 'Stop session'}
            </button>
            {remaining && (
              <button
                type="button"
                className="secondary"
                onClick={() => {
                  touch.mutate()
                }}
              >
                Keep alive
              </button>
            )}
          </div>
        ) : (
          <div className="start">
            <label className="field">
              <span>GPU</span>
              <select
                value={gpu}
                onChange={(e) => {
                  setGpu(e.target.value)
                }}
              >
                {snap.gpus.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={highMem}
                onChange={(e) => {
                  setHighMem(e.target.checked)
                }}
              />{' '}
              High memory
            </label>
            <button
              type="button"
              disabled={start.isPending}
              onClick={() => {
                start.mutate()
              }}
            >
              {start.isPending ? 'Starting…' : 'Start session'}
            </button>
          </div>
        )}
        {error && <p role="alert">{error.message}</p>}
      </section>

      <section className="card">
        <h2>Google Drive</h2>
        <dl className="config">
          <dt>Status</dt>
          <dd>
            {!drive.data?.configured
              ? 'No OAuth client configured'
              : drive.data.authorized
                ? 'Authorized'
                : 'Not authorized: run degas auth drive'}
          </dd>
          {(drive.data?.error ?? snap.drive.push_error) && (
            <>
              <dt>Problem</dt>
              <dd className="error">{drive.data?.error ?? snap.drive.push_error}</dd>
            </>
          )}
          <dt>Last indexed</dt>
          <dd>
            {drive.data?.indexed_at ? new Date(drive.data.indexed_at).toLocaleString() : 'never'}
          </dd>
        </dl>
        <button
          type="button"
          className="secondary"
          disabled={rescan.isPending || !drive.data?.authorized}
          onClick={() => {
            rescan.mutate()
          }}
        >
          {rescan.isPending ? 'Scanning…' : 'Rescan Drive'}
        </button>
        {rescan.error && <p role="alert">{rescan.error.message}</p>}
        {rescan.data && <p className="muted">Indexed {rescan.data.count} assets.</p>}
      </section>

      {(s?.state === 'ready' || s?.state === 'busy') && (
        <section className="card">
          <h2>Troubleshooting</h2>
          <p className="muted small">
            Kills and restarts the worker process on the VM. Loaded models are dropped and the
            running job fails.
          </p>
          <button
            type="button"
            className="secondary"
            disabled={reset.isPending}
            onClick={() => {
              reset.mutate()
            }}
          >
            Force reset worker
          </button>
        </section>
      )}
    </div>
  )
}
