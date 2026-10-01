import type { Asset, Family } from '@/api/types'
import { AssetPicker } from '@/components/AssetPicker/AssetPicker'
import { variantFor } from '@/lib/assets'
import { needsGpu } from '@/lib/gpu'

import { MODE_LABELS } from './modes'

export interface ModelPickerProps {
  family: Family
  /** The families making the same media, whose models are offered too. */
  siblings: Family[]
  mode: string | undefined
  /** The models that can do the mode. */
  models: Asset[]
  selected: string
  onPick: (model: Asset) => void
  onClose: () => void
}

/** Choose the model: any of the same media that can do the mode, whatever its family. */
export function ModelPicker({
  family,
  siblings,
  mode,
  models,
  selected,
  onPick,
  onClose,
}: ModelPickerProps) {
  return (
    <AssetPicker
      title="Model"
      noun="models"
      assets={models}
      selected={new Set([selected])}
      describe={(a) => {
        const f = siblings.find((s) => s.id === a.family)
        const v = f && variantFor(a.path, f)
        const name = v?.model_dir ? v.label : siblings.length > 1 ? (f?.label ?? null) : null
        return [name, v && needsGpu(v.min_gpu)].filter(Boolean).join(', ') || null
      }}
      empty={
        <p>
          No models found for “{MODE_LABELS[mode ?? ''] ?? mode}”. Put{' '}
          {family.media === 'video' ? 'diffusers model folders' : 'checkpoints'} in Drive under{' '}
          <code>
            degas/models/{family.id}/{family.variants.length > 1 ? '<variant>/' : ''}
          </code>
          , then rescan.
        </p>
      }
      onPick={onPick}
      onClose={onClose}
    />
  )
}
