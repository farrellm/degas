import type { Asset, LoraEntry, LoraRef } from '@/api/types'
import { isPair, loraAssets, loraKey, loraLabel } from '@/lib/loras'

export interface LoraListProps {
  loras: LoraEntry[]
  /** The family's LoRAs in the Drive index, for labels and trigger words. */
  index: Asset[] | undefined
  onChange: (loras: LoraEntry[]) => void
  onAdd: () => void
  onTrigger: (word: string) => void
}

/**
 * The LoRAs row in Create: each LoRA with its weight, and trigger words to tap into the
 * prompt. A Wan A14B pair has one weight per expert.
 */
export function LoraList({ loras, index, onChange, onAdd, onTrigger }: LoraListProps) {
  const update = (i: number, lora: LoraEntry | null) =>
    onChange(lora ? loras.with(i, lora) : loras.toSpliced(i, 1))

  return (
    <div className="lora-group" role="group" aria-labelledby="loras-label">
      <div className="setting">
        <span className="setting-label" id="loras-label">
          LoRAs
        </span>
        <button type="button" className="row-action" aria-label="Add LoRA" onClick={onAdd}>
          Add
        </button>
      </div>
      {loras.map((lora, i) => {
        const found = loraAssets(lora, index)
        const label = loraLabel(lora, index)
        const id = `lora-${i}`
        const words = [...new Set(found.flatMap((a) => a?.sidecar?.trigger_words ?? []))]
        const missing = !!index && found.some((a) => !a)
        const weight = (ref: LoraRef, name: string, change: (w: number) => void, half = '') => (
          <>
            <input
              id={`${id}${half}`}
              aria-label={name}
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={ref.weight}
              onChange={(e) => change(Number(e.target.value))}
            />
            <output htmlFor={`${id}${half}`}>{ref.weight.toFixed(2)}</output>
          </>
        )
        return (
          <div key={loraKey(lora)} className={isPair(lora) ? 'lora pair' : 'lora'}>
            <label className="lora-name" htmlFor={isPair(lora) ? `${id}-high` : id}>
              {label}
            </label>
            {isPair(lora) ? (
              <>
                <RemoveButton label={label} onClick={() => update(i, null)} />
                {(['high', 'low'] as const).map((half) => {
                  const ref = lora[half]
                  if (!ref) return null
                  return (
                    <div key={half} className="lora-half">
                      <span aria-hidden>{half === 'high' ? 'High noise' : 'Low noise'}</span>
                      {weight(
                        ref,
                        `${label} ${half}-noise weight`,
                        (w) => update(i, { ...lora, [half]: { ...ref, weight: w } }),
                        `-${half}`,
                      )}
                    </div>
                  )
                })}
              </>
            ) : (
              <>
                {weight(lora, `${label} weight`, (w) => update(i, { ...lora, weight: w }))}
                <RemoveButton label={label} onClick={() => update(i, null)} />
              </>
            )}
            {missing && (
              <p className="lora-missing">Not found in Drive. Remove it, or rescan Drive.</p>
            )}
            {words.length > 0 && (
              <div className="chips">
                {words.map((w) => (
                  <button
                    key={w}
                    type="button"
                    className="chip"
                    aria-label={`Add “${w}” to the prompt`}
                    onClick={() => onTrigger(w)}
                  >
                    <span>{w}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function RemoveButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button type="button" className="lora-remove" aria-label={`Remove ${label}`} onClick={onClick}>
      <svg viewBox="0 0 12 12" aria-hidden>
        <path d="M2 2l8 8M10 2l-8 8" />
      </svg>
    </button>
  )
}
