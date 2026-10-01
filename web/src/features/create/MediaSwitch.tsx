import type { Family } from '@/api/types'

import { MEDIA_LABELS } from './modes'

export interface MediaSwitchProps {
  media: Family['media'][]
  current: Family['media'] | undefined
  onChoose: (media: Family['media']) => void
}

/** Image ⇄ Video, when the server has families for both. */
export function MediaSwitch({ media, current, onChoose }: MediaSwitchProps) {
  if (media.length <= 1) return null
  return (
    <div className="segmented" role="group" aria-label="Make">
      {media.map((m) => (
        <button
          key={m}
          type="button"
          aria-pressed={m === current}
          onClick={() => {
            onChoose(m)
          }}
        >
          {MEDIA_LABELS[m]}
        </button>
      ))}
    </div>
  )
}
