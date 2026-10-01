import type { Asset } from '@/api/types'
import { AssetPicker } from '@/components/AssetPicker/AssetPicker'

import { adapterKind } from './imagePrompt'

const KIND_LABELS = {
  subject: 'Reads the whole picture',
  face: 'Reads faces',
  faceid: 'Reads who a face is (FaceID)',
  composition: 'Reads layout',
} as const

export interface AdapterPickerProps {
  /** The family's image prompt models in the Drive index. */
  adapters: Asset[]
  familyId: string
  selected: string
  /** The family's one model is FLUX.1 Redux rather than a set of IP-Adapters. */
  redux: boolean
  onPick: (adapter: Asset) => void
  onClose: () => void
}

/** Choose the model that reads an image prompt's pictures. */
export function AdapterPicker({
  adapters,
  familyId,
  selected,
  redux,
  onPick,
  onClose,
}: AdapterPickerProps) {
  return (
    <AssetPicker
      title="Image prompt model"
      noun="models"
      assets={adapters}
      selected={new Set([selected])}
      describe={(a) => KIND_LABELS[adapterKind(a)]}
      empty={
        <p>
          No image prompt models found. Put{' '}
          {redux ? 'the FLUX.1-Redux-dev diffusers folder' : 'IP-Adapters'} in Drive under{' '}
          <code>degas/ip_adapters/{familyId}/</code>, then rescan.
        </p>
      }
      onPick={onPick}
      onClose={onClose}
    />
  )
}
