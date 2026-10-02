import type { ImagePromptOptions } from '@/api/types'
import { ChoiceChips } from '@/components/ChoiceChips'
import { detailInfo, DETAILS, type Take, takeInfo } from '@/lib/imagePrompts'

import { FACEID_NOTE, type ImagePromptUnit, takesFor } from './imagePrompt'

export interface TakePickerProps {
  unit: ImagePromptUnit
  /** What the family's image prompts can do. */
  options: ImagePromptOptions
  /** The unit's model reads who a face is (FaceID). */
  faceid: boolean
  onTake: (take: Take) => void
  onDownsample: (downsample: number) => void
}

/** What an image prompt takes from its pictures, and (for Redux) how closely it follows them. */
export function TakePicker({ unit, options, faceid, onTake, onDownsample }: TakePickerProps) {
  const takes = takesFor(options)

  return (
    <>
      {takes.length > 0 && (
        <ChoiceChips
          label="Take from it"
          className="control-traces"
          options={takes}
          value={unit.take}
          onChoose={onTake}
        />
      )}
      {takes.length > 0 && (
        <p className="row-note">
          {unit.take === 'face' && faceid ? FACEID_NOTE : takeInfo(unit.take).note}
        </p>
      )}
      {options.detail && (
        <>
          <ChoiceChips
            label="How closely"
            className="control-traces"
            options={DETAILS.map((d) => ({ id: d.downsample, label: d.label }))}
            value={unit.downsample}
            onChoose={onDownsample}
          />
          <p className="row-note">{detailInfo(unit.downsample).note}</p>
        </>
      )}
    </>
  )
}
