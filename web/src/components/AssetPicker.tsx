import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState, type ReactNode } from 'react'
import { api, thumbUrl, type Asset } from '../api'
import { assetLabel, bytes, copyEstimate, useCachedPaths } from '../assets'
import { ago, useNow } from '../time'
import { Sheet } from './Sheet'

// Below this many rows, a search field is more clutter than help.
const SEARCH_FROM = 7

interface Props {
  title: string
  /** Plural noun for search and empty states: "models", "LoRAs". */
  noun: string
  assets: Asset[]
  selected: Set<string>
  /** Show a preview column even when no row has a preview (LoRAs). */
  thumbs?: boolean
  empty: ReactNode
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
  empty,
  onPick,
  onClose,
}: Props) {
  const [query, setQuery] = useState('')
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
    <Sheet title={title} onClose={onClose}>
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
            const meta = [a.size === null ? null : bytes(a.size), where].filter(Boolean).join(', ')
            return (
              <li key={a.path}>
                <button
                  type="button"
                  className="asset-row"
                  aria-pressed={on}
                  onClick={() => {
                    onPick(a)
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
              </li>
            )
          })}
        </ul>
      )}
      <RescanFooter />
    </Sheet>
  )
}

function RescanFooter() {
  const qc = useQueryClient()
  const now = useNow(60_000)
  const drive = useQuery({ queryKey: ['drive'], queryFn: api.drive })
  const rescan = useMutation({
    mutationFn: api.rescan,
    onSettled: () => qc.invalidateQueries({ queryKey: ['drive'] }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['assets'] }),
  })
  const d = drive.data
  return (
    <div className="asset-footer">
      <p>
        {!d
          ? null
          : !d.authorized
            ? 'Drive isn’t authorized yet.'
            : `Indexed ${d.indexed_at ? ago(d.indexed_at, now) : 'never'}.`}
      </p>
      <button
        type="button"
        className="btn quiet small"
        disabled={rescan.isPending || !d?.authorized}
        onClick={() => {
          rescan.mutate()
        }}
      >
        {rescan.isPending ? 'Rescanning…' : 'Rescan Drive'}
      </button>
      {rescan.error && <p role="alert">{rescan.error.message}</p>}
    </div>
  )
}
