import { useState } from 'react'

import type { Asset, ImagePromptSpec } from '@/api/types'
import { thumbUrl } from '@/api/urls'
import { CoveredText } from '@/components/CoveredText'
import { useCovered } from '@/hooks/useDiscretion'
import { assetLabel } from '@/lib/assets'
import { reveal } from '@/lib/discretion'
import { unref } from '@/lib/image'
import { specSummary } from '@/lib/imagePrompts'

import { PictureViewer } from './PictureViewer'

export interface ImagePromptsProps {
  /** The viewer item's id: its pictures uncover with its prompt. */
  id: string
  prompts: ImagePromptSpec[]
  steps: number
  assets: Asset[] | undefined
  /** The image is uncovered, so these are too. */
  shown: boolean
}

/**
 * The pictures an image was prompted with, under its written prompt on the wall label. A tap
 * opens one larger.
 */
export function ImagePrompts({ id, prompts, steps, assets, shown }: ImagePromptsProps) {
  const covered = useCovered(`prompt:${id}`) && !shown
  // The open picture, counted across the units.
  const [open, setOpen] = useState<number | null>(null)
  // Covering again (leaving the app) closes it.
  if (covered && open !== null) setOpen(null)
  const units = prompts.map((p, i) => ({
    model: assetLabel(p.adapter.path, assets),
    summary: specSummary(p, steps),
    shas: p.images.map(unref),
    first: prompts.slice(0, i).reduce((sum, before) => sum + before.images.length, 0),
  }))
  const pictures = units.flatMap((u) =>
    u.shas.map((sha) => ({ sha, model: u.model, summary: u.summary })),
  )

  return (
    <div
      className={covered ? 'image-prompts covered' : 'image-prompts'}
      role="group"
      aria-label="Image prompts"
    >
      {units.map((unit, i) => (
        // A job's units don't change, so their order is their identity.
        <div key={i} className="image-prompt">
          {/* The stored pictures are the squares the model read. */}
          <span className="image-prompt-pictures">
            {unit.shas.map((sha, n) => {
              const at = unit.first + n
              return (
                <button
                  key={at}
                  type="button"
                  aria-label={
                    covered ? 'Show image prompt' : `Open picture ${String(at + 1)}, ${unit.model}`
                  }
                  onClick={() => {
                    if (covered) reveal(`prompt:${id}`)
                    else setOpen(at)
                  }}
                >
                  <img src={thumbUrl(sha)} alt="" loading="lazy" draggable={false} />
                </button>
              )
            })}
          </span>
          <p className="lines">
            <CoveredText id={`prompt:${id}`} label="Show image prompt" shown={shown}>
              <span>{unit.model}</span>
              <span>{unit.summary}</span>
            </CoveredText>
          </p>
        </div>
      ))}
      {open !== null && (
        <PictureViewer
          pictures={pictures}
          index={open}
          onIndex={setOpen}
          onClose={() => {
            setOpen(null)
          }}
        />
      )}
    </div>
  )
}
