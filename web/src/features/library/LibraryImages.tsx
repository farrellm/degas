import { useInfiniteQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { type CSSProperties, useState } from 'react'

import { api } from '@/api/client'
import { queries, queryKeys } from '@/api/queries'
import { Tile } from '@/components/Tile'
import { Viewer } from '@/components/Viewer/Viewer'
import { draftFromSpec, draftWithSource, useSourceTarget } from '@/features/create/draft'
import { useAssets } from '@/hooks/useAssets'
import { useExtendable } from '@/hooks/useExtendable'
import { useNow } from '@/hooks/useNow'
import { isVideo } from '@/lib/image'

import { asViewerItem, type Kept } from './kept'
import { LibraryActions } from './LibraryActions'
import { TagsField } from './TagsField'

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

export function LibraryImages({ q, onRemix }: { q: string; onRemix: () => void }) {
  const qc = useQueryClient()
  const now = useNow(60_000)
  const assets = useAssets()
  const [open, setOpen] = useState<string | null>(null)
  const sourceTarget = useSourceTarget()
  const extendable = useExtendable()
  const extend = useMutation({
    mutationFn: api.extendLibraryItem,
    onSuccess: (ext) => {
      draftFromSpec(ext.spec, null, ext.source)
      onRemix()
    },
  })
  const library = useInfiniteQuery(queries.library(q))
  const remove = useMutation({
    mutationFn: api.deleteLibraryItem,
    onSettled: () =>
      Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.library }),
        qc.invalidateQueries({ queryKey: queryKeys.results }),
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
        <p>
          Open an image or clip in Results and choose Keep. Kept items stay until you delete them.
        </p>
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
              <Tile
                key={item.id}
                id={item.id}
                blobSha={item.blob_sha}
                mediaType={item.media_type}
                duration={item.duration}
                style={
                  {
                    '--ratio': `${String(item.width ?? 1)} / ${String(item.height ?? 1)}`,
                  } as CSSProperties
                }
                label={`Open ${String(item.config.params.prompt ?? 'image')}`}
                coveredLabel={`Show ${isVideo(item.media_type) ? 'clip' : 'image'}`}
                onOpen={() => {
                  setOpen(item.id)
                }}
              />
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
              error={(remove.error ?? extend.error)?.message}
              onRemix={() => {
                draftFromSpec(item.config, item.seed)
                onRemix()
              }}
              onExtend={
                item.kind === 'video' && extendable(item.config.family)
                  ? () => {
                      extend.mutate(item.id)
                    }
                  : undefined
              }
              onUseAsSource={
                item.kind === 'image' && sourceTarget && item.width && item.height
                  ? () => {
                      draftWithSource(sourceTarget.family, sourceTarget.mode, {
                        sha: item.blob_sha,
                        width: item.width ?? 0,
                        height: item.height ?? 0,
                      })
                      onRemix()
                    }
                  : undefined
              }
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
