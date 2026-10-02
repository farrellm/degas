import type { Asset, ImagePromptSpec } from '@/api/types'
import { thumbUrl } from '@/api/urls'
import { CoveredText } from '@/components/CoveredText'
import { useCovered } from '@/hooks/useDiscretion'
import { assetLabel } from '@/lib/assets'
import { unref } from '@/lib/image'
import { specSummary } from '@/lib/imagePrompts'

export interface ImagePromptsProps {
  /** The viewer item's id: its pictures uncover with its prompt. */
  id: string
  prompts: ImagePromptSpec[]
  steps: number
  assets: Asset[] | undefined
  /** The image is uncovered, so these are too. */
  shown: boolean
}

/** The pictures an image was prompted with, under its written prompt on the wall label. */
export function ImagePrompts({ id, prompts, steps, assets, shown }: ImagePromptsProps) {
  const covered = useCovered(`prompt:${id}`) && !shown
  return (
    <div
      className={covered ? 'image-prompts covered' : 'image-prompts'}
      role="group"
      aria-label="Image prompts"
    >
      {prompts.map((p, i) => (
        // A job's units don't change, so their order is their identity.
        <div key={i} className="image-prompt">
          {/* The stored pictures are the squares the model read. */}
          <span className="image-prompt-pictures">
            {p.images.map((sha, n) => (
              <img key={`${sha}-${String(n)}`} src={thumbUrl(unref(sha))} alt="" loading="lazy" />
            ))}
          </span>
          <p className="lines">
            <CoveredText id={`prompt:${id}`} label="Show image prompt" shown={shown}>
              <span>{assetLabel(p.adapter.path, assets)}</span>
              <span>{specSummary(p, steps)}</span>
            </CoveredText>
          </p>
        </div>
      ))}
    </div>
  )
}
