import type { Family } from '@/api/types'
import { ChoiceChips } from '@/components/ChoiceChips'

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
    <ChoiceChips
      label="Make"
      className="segmented"
      options={media.map((m) => ({ id: m, label: MEDIA_LABELS[m] }))}
      value={current}
      onChoose={onChoose}
    />
  )
}
