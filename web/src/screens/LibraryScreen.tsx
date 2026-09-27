import { useInfiniteQuery, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useDeferredValue, useState, type CSSProperties } from 'react'
import { api, thumbUrl, type LibraryItem, type SavedPrompt } from '../api'
import { useAssets } from '../assets'
import { PromptText } from '../components/PromptSheet'
import { SaveToPhotos, Viewer, type ViewerItem } from '../components/Viewer'
import { draftFromSpec, draftWithPrompt } from '../draft'
import { useNow } from '../time'

type View = 'Images' | 'Prompts'
const VIEWS: View[] = ['Images', 'Prompts']

type Kept = LibraryItem & ViewerItem

function asViewerItem(item: LibraryItem): Kept {
  const seed = item.config.params.seed
  return { ...item, seed: typeof seed === 'number' ? seed : null, spec: item.config }
}

/** "Today", "Yesterday", "Sep 25", or "Sep 25, 2025" in another year. */
function dayLabel(iso: string, now: number): string {
  const d = new Date(iso)
  const today = new Date(now)
  const yesterday = new Date(now - 86_400_000)
  if (d.toDateString() === today.toDateString()) return 'Today'
  if (d.toDateString() === yesterday.toDateString()) return 'Yesterday'
  return d.toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
    ...(d.getFullYear() === today.getFullYear() ? {} : { year: 'numeric' }),
  })
}

export function LibraryScreen({ onRemix }: { onRemix: () => void }) {
  const [view, setView] = useState<View>('Images')
  const [query, setQuery] = useState('')
  const q = useDeferredValue(query.trim())

  return (
    <div className="library">
      <div className="library-bar">
        <input
          type="search"
          className="library-search"
          aria-label="Search the library"
          placeholder="Search prompts and tags"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value)
          }}
        />
        <div className="segmented" role="group" aria-label="Show">
          {VIEWS.map((v) => (
            <button
              key={v}
              type="button"
              aria-pressed={v === view}
              onClick={() => {
                setView(v)
              }}
            >
              {v}
            </button>
          ))}
        </div>
      </div>
      {view === 'Images' ? <Images q={q} onRemix={onRemix} /> : <Prompts q={q} onUse={onRemix} />}
    </div>
  )
}

function Images({ q, onRemix }: { q: string; onRemix: () => void }) {
  const qc = useQueryClient()
  const now = useNow(60_000)
  const assets = useAssets()
  const [open, setOpen] = useState<string | null>(null)
  const library = useInfiniteQuery({
    queryKey: ['library', q],
    queryFn: ({ pageParam }) => api.library(q, pageParam),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (last) => last.cursor ?? undefined,
  })
  const remove = useMutation({
    mutationFn: api.deleteLibraryItem,
    onSettled: () =>
      Promise.all([
        qc.invalidateQueries({ queryKey: ['library'] }),
        qc.invalidateQueries({ queryKey: ['results'] }),
      ]),
  })

  if (library.isPending) return <p className="loading">Loading…</p>
  if (library.error) return <p role="alert">{library.error.message}</p>

  const items = library.data.pages.flatMap((p) => p.items).map(asViewerItem)
  const openIndex = items.findIndex((i) => i.id === open)

  if (items.length === 0) {
    return q ? (
      <p className="library-empty">Nothing kept matches “{q}”.</p>
    ) : (
      <div className="feed-empty">
        <p className="lede">Nothing kept yet.</p>
        <p>Open an image in Results and choose Keep. Kept images stay until you delete them.</p>
      </div>
    )
  }

  const days: { label: string; items: Kept[] }[] = []
  for (const item of items) {
    const label = dayLabel(item.created_at, now)
    const last = days.at(-1)
    if (last?.label === label) last.items.push(item)
    else days.push({ label, items: [item] })
  }

  return (
    <>
      {days.map((day) => (
        <section key={day.label} className="day" aria-label={day.label}>
          <h2>{day.label}</h2>
          <div className="shelf">
            {day.items.map((item) => (
              <button
                key={item.id}
                type="button"
                className="tile"
                style={
                  {
                    '--ratio': `${String(item.width ?? 1)} / ${String(item.height ?? 1)}`,
                  } as CSSProperties
                }
                aria-label={`Open ${String(item.config.params.prompt ?? 'image')}`}
                onClick={() => {
                  setOpen(item.id)
                }}
              >
                <img src={thumbUrl(item.blob_sha)} alt="" loading="lazy" />
              </button>
            ))}
          </div>
        </section>
      ))}
      {library.hasNextPage && (
        <button
          type="button"
          className="btn quiet more-button"
          disabled={library.isFetchingNextPage}
          onClick={() => {
            void library.fetchNextPage()
          }}
        >
          {library.isFetchingNextPage ? 'Loading…' : 'Show older'}
        </button>
      )}
      {openIndex >= 0 && (
        <Viewer
          items={items}
          assets={assets.data}
          index={openIndex}
          onIndex={(i) => {
            setOpen(items[i]?.id ?? null)
          }}
          onClose={() => {
            setOpen(null)
          }}
          extra={(item) => <TagsField key={item.id} item={item} />}
          actions={(item) => (
            <LibraryActions
              item={item}
              deleting={remove.isPending}
              error={remove.error?.message}
              onRemix={() => {
                draftFromSpec(item.config, item.seed)
                onRemix()
              }}
              onDelete={() => {
                const next = items[openIndex + 1] ?? items[openIndex - 1]
                remove.mutate(item.id, {
                  onSuccess: () => {
                    setOpen(next?.id ?? null)
                  },
                })
              }}
            />
          )}
        />
      )}
    </>
  )
}

function LibraryActions({
  item,
  deleting,
  error,
  onRemix,
  onDelete,
}: {
  item: Kept
  deleting: boolean
  error: string | undefined
  onRemix: () => void
  onDelete: () => void
}) {
  const [confirming, setConfirming] = useState(false)
  if (confirming) {
    return (
      <div className="confirm wide" role="group" aria-label="Confirm delete">
        <p>
          Delete this image from the library?
          {item.source_result_id ? '' : ' This can’t be undone.'}
        </p>
        <button type="button" className="btn danger" disabled={deleting} onClick={onDelete}>
          Delete
        </button>
        <button
          type="button"
          className="btn quiet"
          onClick={() => {
            setConfirming(false)
          }}
        >
          Cancel
        </button>
      </div>
    )
  }
  return (
    <>
      <button type="button" className="btn wide" onClick={onRemix}>
        Remix
      </button>
      <SaveToPhotos item={item} />
      <button
        type="button"
        className="btn quiet"
        onClick={() => {
          setConfirming(true)
        }}
      >
        Delete
      </button>
      {error && (
        <p className="viewer-note" role="alert">
          {error}
        </p>
      )}
    </>
  )
}

/** Comma-separated tags, saved when the field loses focus. */
function TagsField({ item }: { item: LibraryItem }) {
  const qc = useQueryClient()
  const [text, setText] = useState(item.tags.join(', '))
  const edit = useMutation({
    mutationFn: (tags: string[]) => api.editLibraryItem(item.id, { tags }),
    onSuccess: (saved) => {
      setText(saved.tags.join(', '))
      return qc.invalidateQueries({ queryKey: ['library'] })
    },
  })
  const commit = () => {
    const tags = text
      .split(',')
      .map((t) => t.trim())
      .filter(Boolean)
    if (tags.join(',') !== item.tags.join(',')) edit.mutate(tags)
  }
  return (
    <div className="tags-field">
      <label htmlFor="tags">Tags</label>
      <input
        id="tags"
        type="text"
        placeholder="Add tags, separated by commas"
        autoCapitalize="none"
        value={text}
        onChange={(e) => {
          setText(e.target.value)
        }}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur()
        }}
      />
      {edit.error && <p role="alert">{edit.error.message}</p>}
    </div>
  )
}

function Prompts({ q, onUse }: { q: string; onUse: () => void }) {
  const prompts = useQuery({ queryKey: ['prompts', q], queryFn: () => api.prompts(q) })

  if (prompts.isPending) return <p className="loading">Loading…</p>
  if (prompts.error) return <p role="alert">{prompts.error.message}</p>
  if (prompts.data.length === 0) {
    return q ? (
      <p className="library-empty">No saved prompts match “{q}”.</p>
    ) : (
      <div className="feed-empty">
        <p className="lede">No saved prompts.</p>
        <p>In Create, open Prompts and choose Save this prompt.</p>
      </div>
    )
  }
  return (
    <ul className="prompt-list library-prompts">
      {prompts.data.map((p) => (
        <PromptRow
          key={p.id}
          p={p}
          onUse={() => {
            draftWithPrompt(p)
            onUse()
          }}
        />
      ))}
    </ul>
  )
}

function PromptRow({ p, onUse }: { p: SavedPrompt; onUse: () => void }) {
  const qc = useQueryClient()
  const [mode, setMode] = useState<'view' | 'rename' | 'delete'>('view')
  const [name, setName] = useState(p.name)
  const onSettled = () => qc.invalidateQueries({ queryKey: ['prompts'] })
  const rename = useMutation({
    mutationFn: () => api.editPrompt(p.id, { name: name.trim() }),
    onSuccess: () => {
      setMode('view')
    },
    onSettled,
  })
  const remove = useMutation({ mutationFn: () => api.deletePrompt(p.id), onSettled })
  const error = rename.error ?? remove.error

  return (
    <li className="prompt-item">
      {mode === 'rename' ? (
        <form
          className="rename"
          onSubmit={(e) => {
            e.preventDefault()
            if (name.trim()) rename.mutate()
          }}
        >
          <input
            aria-label="Prompt name"
            value={name}
            autoFocus
            onChange={(e) => {
              setName(e.target.value)
            }}
          />
          <button type="submit" className="btn small" disabled={!name.trim() || rename.isPending}>
            Rename
          </button>
          <button
            type="button"
            className="btn quiet small"
            onClick={() => {
              setName(p.name)
              setMode('view')
            }}
          >
            Cancel
          </button>
        </form>
      ) : (
        <PromptText p={p} />
      )}
      {mode === 'delete' ? (
        <div className="row-buttons" role="group" aria-label="Confirm delete">
          <span className="confirm-text">Delete this prompt?</span>
          <button
            type="button"
            className="btn danger small"
            disabled={remove.isPending}
            onClick={() => {
              remove.mutate()
            }}
          >
            Delete
          </button>
          <button
            type="button"
            className="btn quiet small"
            onClick={() => {
              setMode('view')
            }}
          >
            Cancel
          </button>
        </div>
      ) : mode === 'view' ? (
        <div className="row-buttons">
          <button type="button" className="btn quiet small" onClick={onUse}>
            Use
          </button>
          <button
            type="button"
            className="btn quiet small"
            aria-label={`Rename ${p.name}`}
            onClick={() => {
              setMode('rename')
            }}
          >
            Rename
          </button>
          <button
            type="button"
            className="btn quiet small"
            aria-label={`Delete ${p.name}`}
            onClick={() => {
              setMode('delete')
            }}
          >
            Delete
          </button>
        </div>
      ) : null}
      {error && <p role="alert">{error.message}</p>}
    </li>
  )
}
