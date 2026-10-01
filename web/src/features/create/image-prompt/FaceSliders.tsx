import { SliderRow } from '@/components/SliderRow'

export interface FaceSlidersProps {
  structure: number
  loraWeight: number
  onStructure: (structure: number) => void
  onLoraWeight: (weight: number) => void
}

/** A FaceID model's two extra weights. */
export function FaceSliders({
  structure,
  loraWeight,
  onStructure,
  onLoraWeight,
}: FaceSlidersProps) {
  return (
    <>
      <SliderRow
        id="prompt-structure"
        label="Face structure"
        min={0}
        max={2}
        value={structure}
        note="How much of the face’s shape comes from the picture, on top of who it is."
        onChange={onStructure}
      />
      <SliderRow
        id="prompt-lora"
        label="Face LoRA"
        min={0}
        max={1.5}
        value={loraWeight}
        note="The LoRA trained with the model. Lower lets the checkpoint’s own look through."
        onChange={onLoraWeight}
      />
    </>
  )
}
