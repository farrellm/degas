import type { Asset } from '@/api/types'
import { AssetPicker } from '@/components/AssetPicker/AssetPicker'

import { controlKind, traceInfo } from './control'

export interface ControlNetPickerProps {
  /** The family's ControlNets in the Drive index. */
  controlnets: Asset[]
  familyId: string
  selected: string
  onPick: (controlnet: Asset) => void
  onClose: () => void
}

/** Choose the ControlNet that reads a unit's control image. */
export function ControlNetPicker({
  controlnets,
  familyId,
  selected,
  onPick,
  onClose,
}: ControlNetPickerProps) {
  return (
    <AssetPicker
      title="ControlNet"
      noun="ControlNets"
      assets={controlnets}
      selected={new Set([selected])}
      describe={(a) => {
        const kind = controlKind(a)
        return kind ? `Reads ${traceInfo(kind).image}` : null
      }}
      empty={
        <p>
          No ControlNets found. Put them in Drive under <code>degas/controlnets/{familyId}/</code>,
          then rescan.
        </p>
      }
      onPick={onPick}
      onClose={onClose}
    />
  )
}
