import { useQuery } from '@tanstack/react-query'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { api, blobUrl, type Asset, type Spec } from '../api'
import { assetLabel } from '../assets'
import { size } from '../format'

/** What the viewer shows: a result from the feed or a kept library item. */
export interface ViewerItem {
  id: string
  blob_sha: string
  media_type: string
  seed: number | null
  width: number | null
  height: number | null
  spec: Spec | null
}

/** "Studio XL v10, with Film Grain v3 at 0.8" for the wall label. */
function modelWithLoras(spec: Spec, assets: Asset[] | undefined): string {
  const model = assetLabel(spec.model.path, assets)
  const loras = (spec.loras ?? []).map(
    (l) => `${assetLabel(l.path, assets)} at ${String(Number(l.weight.toFixed(2)))}`,
  )
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
  const sampler = schema.data?.properties.scheduler
  const samplerIndex = sampler?.enum?.indexOf(String(params.scheduler)) ?? -1
  const samplerLabel = sampler?.['x-enum-labels']?.[samplerIndex] ?? String(params.scheduler)

  return (
    <div
      ref={ref}
      className="viewer"
      role="dialog"
      aria-modal="true"
      aria-label="Image"
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
        <img src={blobUrl(r.blob_sha)} alt={String(params.prompt ?? '')} draggable={false} />
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
          <span>
            {String(params.steps)} steps, CFG {String(params.cfg)}, {samplerLabel}
          </span>
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
    const file = new File([blob], `degas-${String(item.seed)}.png`, { type: item.media_type })
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
