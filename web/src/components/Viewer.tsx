import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { api, blobUrl, isPair, isVideo, type Asset, type SavedConfig, type Spec } from '../api'
import { assetLabel, loraLabel } from '../assets'
import { duration, size } from '../format'
import { enumLabel } from '../schema'

/** What the viewer shows: a result from the feed or a kept library item. */
export interface ViewerItem {
  id: string
  blob_sha: string
  media_type: string
  seed: number | null
  width: number | null
  height: number | null
  duration?: number | null
  /** A stitched video extension: its clips' configs. */
  segments?: SavedConfig[] | null
  spec: Spec | null
}

const weight = (w: number) => String(Number(w.toFixed(2)))

/** "Studio XL v10, with Film Grain v3 at 0.8" for the wall label. */
function modelWithLoras(spec: Spec, assets: Asset[] | undefined): string {
  const model = assetLabel(spec.model.path, assets)
  const loras = (spec.loras ?? []).map((l) => {
    const at = isPair(l)
      ? [l.high && `${weight(l.high.weight)} high`, l.low && `${weight(l.low.weight)} low`]
          .filter(Boolean)
          .join(', ')
      : weight(l.weight)
    return `${loraLabel(l, assets)} at ${at}`
  })
  return loras.length ? `${model}, with ${loras.join(' and ')}` : model
}

interface Props<T extends ViewerItem> {
  items: T[]
  assets: Asset[] | undefined
  index: number
  onIndex: (i: number) => void
  onClose: () => void
  /** The wall label's buttons for the current item. */
  actions: (item: T) => ReactNode
  /** Extra wall-label content under the settings lines (e.g. tags). */
  extra?: (item: T) => ReactNode
}

/** Full-screen image on the well, with its settings as a wall label. */
export function Viewer<T extends ViewerItem>({
  items,
  assets,
  index,
  onIndex,
  onClose,
  actions,
  extra,
}: Props<T>) {
  const r = items[index]
  const ref = useRef<HTMLDivElement>(null)
  const swipe = useRef<number | null>(null)

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    ref.current?.focus()
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = overflow
      opener?.focus()
    }
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowRight' && index < items.length - 1) onIndex(index + 1)
      if (e.key === 'ArrowLeft' && index > 0) onIndex(index - 1)
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
    }
  }, [index, items.length, onIndex, onClose])

  const spec = r?.spec
  const schema = useQuery({
    queryKey: ['schema', spec?.family, spec?.variant, spec?.mode],
    queryFn: () => api.schema(spec?.family ?? '', spec?.variant ?? '', spec?.mode ?? ''),
    enabled: !!spec,
    staleTime: Infinity,
  })

  if (!r) return null
  const params = spec?.params ?? {}
  const props = schema.data?.properties
  const samplerLabel = [
    enumLabel(props?.scheduler, params.scheduler),
    params.schedule === 'default' ? null : enumLabel(props?.schedule, params.schedule),
  ]
    .filter(Boolean)
    .join(' ')
  const video = isVideo(r.media_type)
  const cfg =
    params.cfg_low == null
      ? String(params.cfg)
      : `${String(params.cfg)} / ${String(params.cfg_low)}`
  const sampling = [
    `${String(params.steps)} steps`,
    `CFG ${cfg}`,
    samplerLabel,
    params.vae_fp32 === true ? 'float32 VAE' : null,
  ].filter(Boolean)
  const frames = Number(params.num_frames)
  const fps = Number(params.fps)
  const length = r.duration ?? (frames && fps ? frames / fps : null)

  return (
    <div
      ref={ref}
      className="viewer"
      role="dialog"
      aria-modal="true"
      aria-label={video ? 'Video' : 'Image'}
      tabIndex={-1}
    >
      <div className="viewer-bar">
        <button type="button" className="btn quiet small" onClick={onClose}>
          Close
        </button>
        <span className="position">
          {index + 1} of {items.length}
        </span>
        <span className="sheet-actions">
          <button
            type="button"
            className="btn quiet small"
            aria-label="Previous image"
            disabled={index === 0}
            onClick={() => {
              onIndex(index - 1)
            }}
          >
            ‹
          </button>
          <button
            type="button"
            className="btn quiet small"
            aria-label="Next image"
            disabled={index === items.length - 1}
            onClick={() => {
              onIndex(index + 1)
            }}
          >
            ›
          </button>
        </span>
      </div>
      <div
        className="viewer-image"
        onPointerDown={(e) => {
          swipe.current = e.clientX
        }}
        onPointerUp={(e) => {
          if (swipe.current === null) return
          const dx = e.clientX - swipe.current
          swipe.current = null
          if (dx < -50 && index < items.length - 1) onIndex(index + 1)
          if (dx > 50 && index > 0) onIndex(index - 1)
        }}
      >
        {video ? (
          <video
            key={r.blob_sha}
            src={blobUrl(r.blob_sha)}
            aria-label={String(params.prompt ?? 'Video')}
            controls
            autoPlay
            loop
            muted
            playsInline
          />
        ) : (
          <img src={blobUrl(r.blob_sha)} alt={String(params.prompt ?? '')} draggable={false} />
        )}
      </div>
      <div className="wall-label">
        <div>
          <p className="title">{String(params.prompt ?? '') || 'No prompt'}</p>
          {params.negative_prompt ? (
            <p className="avoid">Negative: {String(params.negative_prompt)}</p>
          ) : null}
        </div>
        <p className="lines">
          <span>{spec ? modelWithLoras(spec, assets) : 'Unknown model'}</span>
          <span>
            {size(r.width, r.height)}, seed {r.seed}
          </span>
          {video && r.segments ? (
            <span>
              Extended, {r.segments.length} clips{length ? `, ${duration(length)}` : ''}
            </span>
          ) : video ? (
            <span>
              {frames} frames at {fps} fps{length ? `, ${duration(length)}` : ''}
            </span>
          ) : null}
          <span>{sampling.join(', ')}</span>
        </p>
        {extra?.(r)}
        {/* Keyed so per-item state (errors, confirmations) resets on swipe. */}
        <div key={r.id} className="viewer-actions">
          {actions(r)}
        </div>
      </div>
    </div>
  )
}

const EXTENSIONS: Record<string, string> = {
  'image/png': 'png',
  'image/jpeg': 'jpg',
  'image/webp': 'webp',
  'video/mp4': 'mp4',
  'video/webm': 'webm',
  'video/quicktime': 'mov',
}

/** Export to the phone: the share sheet where there is one, else a download. */
export function SaveToPhotos({
  item,
  className = 'btn quiet',
}: {
  item: ViewerItem
  className?: string
}) {
  const [error, setError] = useState<string | null>(null)

  const share = async () => {
    setError(null)
    const blob = await (await fetch(blobUrl(item.blob_sha))).blob()
    const ext = EXTENSIONS[item.media_type] ?? 'png'
    const file = new File([blob], `degas-${String(item.seed)}.${ext}`, { type: item.media_type })
    if ('canShare' in navigator && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file] })
    } else {
      const a = document.createElement('a')
      a.href = blobUrl(item.blob_sha)
      a.download = file.name
      a.click()
    }
  }

  return (
    <>
      <button
        type="button"
        className={className}
        onClick={() => {
          void share().catch((e: unknown) => {
            if (!(e instanceof DOMException && e.name === 'AbortError')) {
              setError('Saving failed. Try again, or long-press the image.')
            }
          })
        }}
      >
        Save to Photos
      </button>
      {error && (
        <p className="viewer-note" role="alert">
          {error}
        </p>
      )}
    </>
  )
}
