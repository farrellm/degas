import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '@/api/client'
import { queries, queryKeys } from '@/api/queries'
import type { SessionSnapshot } from '@/api/types'
import { Sheet } from '@/components/Sheet'
import { loadDraft, useDraftVariant } from '@/features/create/draft'
import { useNow } from '@/hooks/useNow'
import { belowGpu, GiB, GPU_VRAM, GPUS } from '@/lib/gpu'
import { isActive } from '@/lib/session'
import { countdown } from '@/lib/time'

import { CacheSection } from './CacheSection'
import { DriveSection } from './DriveSection'
import { lastGpu, rememberGpu } from './gpuPreference'
import { NotificationsSection } from './NotificationsSection'

/** Whether the model in the Create draft is a Wan A14B variant. */
function wantsHighMem(): boolean {
  const draft = loadDraft()
  return /\/[^/]*a14b(\/|$)/i.test(draft.families[draft.family]?.model ?? '')
}

/** GPU session, Drive and troubleshooting, opened from the header chip. */
export function SessionSheet({ onClose }: { onClose: () => void }) {
  const session = useQuery(queries.session())
  return (
    <Sheet title="GPU session" onClose={onClose}>
      {session.error && <p role="alert">{session.error.message}</p>}
      {session.data && <SessionBody snap={session.data} />}
      <NotificationsSection />
      <DriveSection authorizedHint={session.data?.drive} />
    </Sheet>
  )
}

function SessionBody({ snap }: { snap: SessionSnapshot }) {
  const qc = useQueryClient()
  const now = useNow(1000)
  const variant = useDraftVariant()
  // The model open in Create sets a floor: GPUs below it are dimmed, and the default rises to it.
  const min = variant && GPUS.indexOf(variant.min_gpu) > 0 ? variant.min_gpu : null
  const [chosen, setGpu] = useState<string | null>(null)
  const last = lastGpu()
  const gpu =
    chosen ??
    (min && belowGpu(last, min) ? (snap.gpus.find((g) => !belowGpu(g, min)) ?? last) : last)
  const short = !!min && belowGpu(gpu, min)
  // Wan A14B needs more than the standard 12 GB of system RAM (design §3.1).
  const [highMem, setHighMem] = useState(wantsHighMem)
  const onSettled = () => qc.invalidateQueries({ queryKey: queryKeys.session })
  const start = useMutation({
    mutationFn: () => {
      rememberGpu(gpu)
      return api.startSession(gpu, highMem)
    },
    onSettled,
  })
  const stop = useMutation({ mutationFn: api.stopSession, onSettled })
  const touch = useMutation({ mutationFn: api.touchSession, onSettled })
  const reset = useMutation({ mutationFn: api.resetWorker, onSettled })

  const s = snap.session
  const active = isActive(snap)
  const idle = countdown(snap.idle_deadline, now)
  const error = start.error ?? stop.error ?? reset.error ?? touch.error
  const w = snap.worker

  if (!active) {
    return (
      <section className="sheet-section" aria-label="Start a session">
        {s?.state === 'error' && s.error && (
          <p className="problem">Last session failed: {s.error}</p>
        )}
        <div className="gpu-choice" role="group" aria-label="GPU">
          {snap.gpus.map((g) => (
            <button
              key={g}
              type="button"
              aria-pressed={g === gpu}
              aria-describedby={min && belowGpu(g, min) ? 'gpu-floor' : undefined}
              className={min && belowGpu(g, min) ? 'short' : undefined}
              onClick={() => {
                setGpu(g)
              }}
            >
              <span className="name">{g}</span>
              {GPU_VRAM[g] && <span className="vram">{GPU_VRAM[g]}</span>}
            </button>
          ))}
        </div>
        {min && variant && (
          <p id="gpu-floor" className={short ? 'warn' : undefined}>
            {short
              ? `${variant.label} needs an ${min}; ${gpu === 'T4' ? 'a' : 'an'} ${gpu} may run it slowly.`
              : `${variant.label} needs an ${min} or better.`}
          </p>
        )}
        <label className="switch">
          <span>
            High memory
            <small>More system RAM. Needed for large video models.</small>
          </span>
          <input
            type="checkbox"
            checked={highMem}
            onChange={(e) => {
              setHighMem(e.target.checked)
            }}
          />
        </label>
        <button
          type="button"
          className="btn"
          disabled={start.isPending}
          onClick={() => {
            start.mutate()
          }}
        >
          {start.isPending ? 'Starting…' : `Start ${gpu} session`}
        </button>
        <p>Stops by itself after {snap.idle_timeout_min} minutes without jobs.</p>
        {error && <p role="alert">{error.message}</p>}
      </section>
    )
  }

  const vramUsed = w?.vram_total != null && w.vram_free != null ? w.vram_total - w.vram_free : null
  return (
    <>
      <section className="sheet-section" aria-label="Current session">
        <div className="status-line">
          <span className={`dot ${s?.state ?? ''}`} aria-hidden />
          {s?.state === 'starting'
            ? `Starting ${s.gpu}${snap.step ? `: ${snap.step}` : ''}`
            : `${s?.state === 'busy' ? 'Generating' : 'Ready'} on ${w?.gpu ?? s?.gpu ?? 'GPU'}`}
        </div>
        {vramUsed !== null && w?.vram_total != null && (
          <div>
            <div className="meter" aria-hidden>
              <span style={{ width: `${String((100 * vramUsed) / w.vram_total)}%` }} />
            </div>
            <p className="bar-note">
              {(vramUsed / GiB).toFixed(1)} of {(w.vram_total / GiB).toFixed(1)} GiB VRAM in use
            </p>
          </div>
        )}
        {idle && s?.state === 'ready' && (
          <p>
            {s.gpu} stops in {idle} if no job runs.
          </p>
        )}
        <div className="sheet-actions">
          <button
            type="button"
            className="btn danger"
            disabled={stop.isPending}
            onClick={() => {
              stop.mutate()
            }}
          >
            {stop.isPending ? 'Stopping…' : 'Stop session'}
          </button>
          {idle && (
            <button
              type="button"
              className="btn quiet"
              onClick={() => {
                touch.mutate()
              }}
            >
              Keep running
            </button>
          )}
        </div>
        {error && <p role="alert">{error.message}</p>}
      </section>
      {w?.cache && <CacheSection cache={w.cache} />}
      {(s?.state === 'ready' || s?.state === 'busy') && (
        <section className="sheet-section" aria-label="Troubleshooting">
          <h3>Worker stuck?</h3>
          <p>
            Restarting the worker drops loaded models, and the running job fails. Queued jobs stay
            queued.
          </p>
          <div className="sheet-actions">
            <button
              type="button"
              className="btn quiet"
              disabled={reset.isPending}
              onClick={() => {
                reset.mutate()
              }}
            >
              {reset.isPending ? 'Restarting…' : 'Restart worker'}
            </button>
          </div>
        </section>
      )}
    </>
  )
}
