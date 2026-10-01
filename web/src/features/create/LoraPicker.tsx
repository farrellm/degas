import { api } from '@/api/client'
import type { Asset, Family, LoraEntry, Variant } from '@/api/types'
import { AssetPicker } from '@/components/AssetPicker/AssetPicker'
import { isPair, loraChoices, loraPaths, sameLora } from '@/lib/loras'

import { CivitaiImport } from './CivitaiImport'

export interface LoraPickerProps {
  family: Family | undefined
  variant: Variant | undefined
  /** The family's LoRAs in the Drive index. */
  index: Asset[] | undefined
  /** The LoRAs already in the form. */
  loras: LoraEntry[]
  onAdd: (lora: LoraEntry) => void
  /** These files went to Drive's trash. */
  onDeleted: (paths: string[]) => void
  onClose: () => void
}

/** Add a LoRA that fits the model's variant, import one from Civitai, or delete one from Drive. */
export function LoraPicker({
  family,
  variant,
  index,
  loras,
  onAdd,
  onDeleted,
  onClose,
}: LoraPickerProps) {
  const choices = loraChoices(index, variant)

  return (
    <AssetPicker
      title="Add LoRA"
      noun="LoRAs"
      assets={choices.rows}
      selected={
        new Set(
          choices.rows
            .filter((row) => loras.some((l) => sameLora(l, choices.entryFor(row))))
            .map((row) => row.path),
        )
      }
      thumbs
      empty={
        <p>
          No LoRAs for this model. Put them in Drive under <code>degas/loras/{family?.id}/</code>,
          then rescan.
          {variant?.lora_format === 'paired_hi_lo' &&
            ' A14B LoRAs come in pairs named …_high_noise and …_low_noise.'}
        </p>
      }
      footer={
        family && (
          <CivitaiImport
            family={family.id}
            onImported={(paths, fresh) => {
              const added = loraChoices(
                fresh.filter((a) => a.family === family.id && a.kind === 'lora'),
                variant,
              )
              const row = added.rows.find((r) =>
                loraPaths(added.entryFor(r)).some((p) => paths.includes(p)),
              )
              if (!row) return false
              onAdd(added.entryFor(row))
              return true
            }}
          />
        )
      }
      deleting={{
        note: (row) =>
          isPair(choices.entryFor(row))
            ? 'Both halves go to Drive’s trash.'
            : 'Its files go to Drive’s trash.',
        run: async (row) => {
          const paths = loraPaths(choices.entryFor(row))
          await api.deleteLoras(paths)
          onDeleted(paths)
        },
      }}
      onPick={(row) => {
        onAdd(choices.entryFor(row))
        onClose()
      }}
      onClose={onClose}
    />
  )
}
