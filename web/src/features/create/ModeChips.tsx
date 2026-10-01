import { ChoiceChips } from '@/components/ChoiceChips'

import { MODE_LABELS } from './modes'

export interface ModeChipsProps {
  modes: string[]
  mode: string | undefined
  onChoose: (mode: string) => void
}

/** What the job starts from: text, an image, an edit… */
export function ModeChips({ modes, mode, onChoose }: ModeChipsProps) {
  return (
    <ChoiceChips
      label="Start from"
      className="mode-chips"
      options={modes.map((m) => ({ id: m, label: MODE_LABELS[m] ?? m }))}
      value={mode}
      onChoose={onChoose}
    />
  )
}
