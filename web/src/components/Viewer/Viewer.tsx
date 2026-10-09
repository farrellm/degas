import { useQuery } from '@tanstack/react-query'
import { type ReactNode, useCallback, useEffect, useRef } from 'react'

import { queries } from '@/api/queries'
import type { Asset, SavedConfig, Spec } from '@/api/types'
import { blobUrl } from '@/api/urls'
import { CoveredText } from '@/components/CoveredText'
import { useDialog } from '@/hooks/useDialog'
import { useCovered } from '@/hooks/useDiscretion'
import { assetLabel } from '@/lib/assets'
import { coverAll, reveal } from '@/lib/discretion'
import { formatDuration, formatSize } from '@/lib/format'
import { isVideo } from '@/lib/image'
import { isPair, loraLabel } from '@/lib/loras'
import { enumLabel } from '@/lib/schema'

import { ImagePrompts } from './ImagePrompts'

/** What the viewer shows: a result from the feed or a kept library item. */
export interface ViewerItem {
  id: string
  /** Null while it's still being made: the viewer draws `sketch` instead. */
  blob_sha: string | null
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

/** "Film Grain v3 at 0.8 and …" for the wall label's model line, or null without LoRAs. */
function loraText(spec: Spec, assets: Asset[] | undefined): string | null {
  const loras = (spec.loras ?? []).map((l) => {
    const at = isPair(l)
      ? [l.high && `${weight(l.high.weight)} high`, l.low && `${weight(l.low.weight)} low`]
          .filter(Boolean)
          .join(', ')
      : weight(l.weight)
    return `${loraLabel(l, assets)} at ${at}`
  })
  return loras.length ? loras.join(' and ') : null
}

export interface ViewerProps<T extends ViewerItem> {
  items: T[]
  assets: Asset[] | undefined
  index: number
  onIndex: (i: number) => void
  onClose: () => void
  /** The wall label's buttons for the current item. */
  actions: (item: T) => ReactNode
  /** Extra wall-label content under the settings lines (e.g. tags). */
  extra?: (item: T) => ReactNode
  /** What stands in for an item that has no picture yet. */
  sketch?: (item: T) => ReactNode
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
  sketch,
}: ViewerProps<T>) {
  const r = items[index]
  const ref = useDialog()
  const swipe = useRef<number | null>(null)
  // A swipe ends in a click on the cover, which mustn't uncover the next image.
  const swiped = useRef(false)
  const covered = useCovered(r?.id ?? '')
  // Leaving the viewer covers everything again.
  const close = useCallback(() => {
    coverAll()
    onClose()
  }, [onClose])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      if (e.key === 'Escape') close()
      if (e.key === 'ArrowRight' && index < items.length - 1) onIndex(index + 1)
      if (e.key === 'ArrowLeft' && index > 0) onIndex(index - 1)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [index, items.length, onIndex, close])

  const spec = r?.spec
  const schema = useQuery(queries.schema(spec?.family, spec?.variant, spec?.mode))

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
  const loras = spec ? loraText(spec, assets) : null
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
        <button type="button" className="btn quiet small" onClick={close}>
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
            onClick={() => onIndex(index - 1)}
          >
            ‹
          </button>
          <button
            type="button"
            className="btn quiet small"
            aria-label="Next image"
            disabled={index === items.length - 1}
            onClick={() => onIndex(index + 1)}
          >
            ›
          </button>
        </span>
      </div>
      <div
        className={covered ? 'viewer-image covered' : 'viewer-image'}
        onPointerDown={(e) => {
          swipe.current = e.clientX
        }}
        onPointerUp={(e) => {
          if (swipe.current === null) return
          const dx = e.clientX - swipe.current
          swipe.current = null
          swiped.current = Math.abs(dx) > 10
          if (dx < -50 && index < items.length - 1) onIndex(index + 1)
          if (dx > 50 && index > 0) onIndex(index - 1)
        }}
      >
        {r.blob_sha === null ? (
          sketch?.(r)
        ) : video ? (
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
        {covered && r.blob_sha !== null && (
          <button
            type="button"
            className="cover-button"
            aria-label={video ? 'Show clip' : 'Show image'}
            onClick={() => {
              if (swiped.current) swiped.current = false
              else reveal(r.id)
            }}
          />
        )}
      </div>
      <div className="wall-label">
        <div>
          <p className="title">
            <CoveredText id={`prompt:${r.id}`} shown={!covered}>
              {String(params.prompt ?? '') || 'No prompt'}
            </CoveredText>
          </p>
          {params.negative_prompt ? (
            <p className="avoid">
              <CoveredText id={`prompt:${r.id}`} label="Show negative prompt" shown={!covered}>
                Negative: {String(params.negative_prompt)}
              </CoveredText>
            </p>
          ) : null}
          {spec?.image_prompts?.length ? (
            <ImagePrompts
              key={r.id}
              id={r.id}
              prompts={spec.image_prompts}
              steps={Number(params.steps)}
              assets={assets}
              shown={!covered}
            />
          ) : null}
        </div>
        <p className="lines">
          <span>
            {spec ? (
              <CoveredText id={`prompt:${r.id}`} label="Show model" shown={!covered}>
                {assetLabel(spec.model.path, assets)}
                {loras && `, with ${loras}`}
              </CoveredText>
            ) : (
              'Unknown model'
            )}
          </span>
          <span>
            {formatSize(r.width, r.height)}, seed {r.seed}
          </span>
          {video && r.segments ? (
            <span>
              Extended, {r.segments.length} clips{length ? `, ${formatDuration(length)}` : ''}
            </span>
          ) : video ? (
            <span>
              {frames} frames at {fps} fps{length ? `, ${formatDuration(length)}` : ''}
            </span>
          ) : null}
          <span>{sampling.join(', ')}</span>
        </p>
        {extra?.(r)}
        {/* Keyed so per-item state (errors, confirmations) resets on swipe; prefixed so it can't
            collide with a keyed `extra` beside it. */}
        <div key={`actions:${r.id}`} className="viewer-actions">
          {actions(r)}
        </div>
      </div>
    </div>
  )
}
