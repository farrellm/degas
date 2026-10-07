import { useMutation, useQueryClient } from '@tanstack/react-query'
import { type ReactNode, useState } from 'react'

import { queryKeys } from '@/api/queries'
import type { Asset } from '@/api/types'
import { thumbUrl } from '@/api/urls'
import { Sheet } from '@/components/Sheet'
import { useCachedPaths } from '@/hooks/useAssets'
import { assetLabel, copyEstimate } from '@/lib/assets'
import { formatBytes } from '@/lib/format'

import { RescanFooter } from './RescanFooter'

// Below this many rows, a search field is more clutter than help.
const SEARCH_FROM = 7

export interface AssetPickerProps {
  title: string
  /** Plural noun for search and empty states: "models", "LoRAs". */
  noun: string
  assets: Asset[]
  selected: Set<string>
  /** Show a preview column even when no row has a preview (LoRAs). */
  thumbs?: boolean
  /** A first fact for each row's meta line (e.g. the model's variant). */
  describe?: (asset: Asset) => string | null
  empty: ReactNode
  /** Above the rescan footer (the LoRA picker's Import a LoRA). */
  footer?: ReactNode
  /** Offer a delete mode: `note` follows the question, `run` deletes the row's files. */
  deleting?: {
    note: (asset: Asset) => string
    run: (asset: Asset) => Promise<unknown>
  }
  onPick: (asset: Asset) => void
  onClose: () => void
}

/** Choose a model or LoRA from the Drive index, with what each costs to bring to the GPU. */
export function AssetPicker({
  title,
  noun,
  assets,
  selected,
  thumbs = false,
  describe,
  empty,
  footer,
  deleting,
  onPick,
  onClose,
}: AssetPickerProps) {
  const [query, setQuery] = useState('')
  const [deleteMode, setDeleteMode] = useState(false)
  const [confirming, setConfirming] = useState<string | null>(null)
  const [deleted, setDeleted] = useState<string | null>(null)
  const qc = useQueryClient()
  const remove = useMutation({
    mutationFn: (a: Asset) => deleting?.run(a) ?? Promise.resolve(),
    onSuccess: (_, a) => {
      setDeleted(`Deleted ${assetLabel(a.path, assets)}.`)
      setConfirming(null)
    },
    onSettled: () => qc.invalidateQueries({ queryKey: queryKeys.assets }),
  })
  const cached = useCachedPaths()
  const q = query.trim().toLowerCase()
  const shown = q
    ? assets.filter((a) =>
        [assetLabel(a.path, assets), a.path, ...(a.sidecar?.trigger_words ?? [])].some((s) =>
          s.toLowerCase().includes(q),
        ),
      )
    : assets
  const withThumbs = thumbs || assets.some((a) => a.preview_thumb)

  return (
    <Sheet
      title={deleteMode ? `Delete ${noun}` : title}
      actions={
        deleting &&
        assets.length > 0 && (
          <button
            type="button"
            className="btn quiet small"
            aria-pressed={deleteMode}
            onClick={() => {
              setDeleteMode((on) => !on)
              setConfirming(null)
              setDeleted(null)
              remove.reset()
            }}
          >
            {deleteMode ? 'Stop deleting' : `Delete ${noun}`}
          </button>
        )
      }
      onClose={onClose}
    >
      {assets.length >= SEARCH_FROM && (
        <input
          type="search"
          className="asset-search"
          aria-label={`Search ${noun}`}
          placeholder={`Search ${noun}`}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
          }}
        />
      )}
      {assets.length === 0 ? (
        <div className="asset-empty">{empty}</div>
      ) : shown.length === 0 ? (
        <p className="asset-empty">
          No {noun} match “{query.trim()}”.
        </p>
      ) : (
        <ul className="asset-list">
          {shown.map((a) => {
            const on = selected.has(a.path)
            const where =
              a.size === null || cached === null
                ? null
                : cached.has(a.path)
                  ? 'on the GPU'
                  : copyEstimate(a.size)
            const meta = [describe?.(a), a.size === null ? null : formatBytes(a.size), where]
              .filter(Boolean)
              .join(', ')
            const asking = deleteMode && confirming === a.path
            const going = remove.isPending && remove.variables.path === a.path
            return (
              <li key={a.path} className={asking || going ? 'doomed' : undefined}>
                <button
                  type="button"
                  className={deleteMode ? 'asset-row deleting' : 'asset-row'}
                  aria-pressed={deleteMode ? undefined : on}
                  aria-expanded={deleteMode ? asking : undefined}
                  disabled={remove.isPending}
                  onClick={() => {
                    if (!deleteMode) {
                      onPick(a)
                      return
                    }
                    setConfirming(asking ? null : a.path)
                    setDeleted(null)
                    remove.reset()
                  }}
                >
                  {withThumbs && (
                    <span className="asset-thumb" aria-hidden>
                      {a.preview_thumb && <img src={thumbUrl(a.preview_thumb)} alt="" />}
                    </span>
                  )}
                  <span className="asset-text">
                    <span className="asset-name">{assetLabel(a.path, assets)}</span>
                    {meta && <span className="asset-meta">{meta}</span>}
                    {a.sidecar?.notes && <span className="asset-notes">{a.sidecar.notes}</span>}
                  </span>
                  <span className="asset-check" aria-hidden />
                </button>
                {(asking || going) && deleting && (
                  <div className="confirm asset-confirm" role="group" aria-label="Confirm delete">
                    <p>
                      Delete {assetLabel(a.path, assets)}? {deleting.note(a)}
                    </p>
                    <button
                      type="button"
                      className="btn danger"
                      disabled={going}
                      onClick={() => {
                        remove.mutate(a)
                      }}
                    >
                      {going ? 'Deleting…' : 'Delete'}
                    </button>
                    <button
                      type="button"
                      className="btn quiet"
                      disabled={going}
                      onClick={() => {
                        setConfirming(null)
                        remove.reset()
                      }}
                    >
                      Cancel
                    </button>
                    {remove.error && <p role="alert">{remove.error.message}</p>}
                  </div>
                )}
              </li>
            )
          })}
        </ul>
      )}
      <p className="asset-deleted" aria-live="polite">
        {deleted}
      </p>
      {!deleteMode && footer}
      <RescanFooter />
    </Sheet>
  )
}
