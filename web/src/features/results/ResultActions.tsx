import type { Result } from '@/api/types'
import { SavePrompt } from '@/components/Viewer/SavePrompt'
import { SaveToPhotos } from '@/components/Viewer/SaveToPhotos'
import { draftFromSpec, draftWithSource, useSourceTarget } from '@/features/create/draft'
import { useExtendable } from '@/hooks/useExtendable'
import { isVideo } from '@/lib/image'

export interface ResultActionsProps {
  result: Result
  keeping: boolean
  extending: boolean
  /** What went wrong keeping or extending it. */
  error: string | undefined
  onKeep: () => void
  onExtend: () => void
  /** The Create draft was replaced: go there. */
  onRemix: () => void
}

/** What the viewer offers for a result: keep it, save it, and start again from it. */
export function ResultActions({
  result: r,
  keeping,
  extending,
  error,
  onKeep,
  onExtend,
  onRemix,
}: ResultActionsProps) {
  const sourceTarget = useSourceTarget()
  const extendable = useExtendable()

  return (
    <>
      <button
        type="button"
        className={r.library_id ? 'btn quiet kept wide' : 'btn wide'}
        aria-pressed={!!r.library_id}
        disabled={keeping}
        onClick={onKeep}
      >
        {r.library_id ? 'Kept' : 'Keep'}
      </button>
      <SaveToPhotos item={r} />
      <SavePrompt item={r} />
      <button
        type="button"
        className="btn quiet"
        disabled={!r.spec}
        onClick={() => {
          if (r.spec) draftFromSpec(r.spec, r.seed)
          onRemix()
        }}
      >
        Remix
      </button>
      {isVideo(r.media_type) ? (
        <button
          type="button"
          className="btn quiet"
          disabled={extending || !extendable(r.spec?.family)}
          onClick={onExtend}
        >
          {extending ? 'Extending…' : 'Extend'}
        </button>
      ) : (
        <button
          type="button"
          className="btn quiet"
          disabled={!sourceTarget || r.width === null || r.height === null}
          onClick={() => {
            if (!sourceTarget || r.width === null || r.height === null) return
            draftWithSource(sourceTarget.family, sourceTarget.mode, {
              sha: r.blob_sha,
              width: r.width,
              height: r.height,
            })
            onRemix()
          }}
        >
          Use as source
        </button>
      )}
      {error && (
        <p className="viewer-note" role="alert">
          {error}
        </p>
      )}
    </>
  )
}
