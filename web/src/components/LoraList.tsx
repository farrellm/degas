import type { Asset, LoraRef } from '../api'
import { assetLabel } from '../assets'

interface Props {
  loras: LoraRef[]
  /** The family's LoRAs in the Drive index, for labels and trigger words. */
  index: Asset[] | undefined
  onChange: (loras: LoraRef[]) => void
  onAdd: () => void
  onTrigger: (word: string) => void
}

/** The LoRAs row in Create: each LoRA with its weight, and trigger words to tap into the prompt. */
export function LoraList({ loras, index, onChange, onAdd, onTrigger }: Props) {
  const update = (i: number, lora: LoraRef | null) => {
    onChange(lora ? loras.with(i, lora) : loras.toSpliced(i, 1))
  }

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
        const asset = index?.find((a) => a.path === lora.path)
        const label = assetLabel(lora.path, index)
        const id = `lora-${String(i)}`
        const words = asset?.sidecar?.trigger_words ?? []
        return (
          <div key={lora.path} className="lora">
            <label className="lora-name" htmlFor={id}>
              {label}
            </label>
            <input
              id={id}
              aria-label={`${label} weight`}
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={lora.weight}
              onChange={(e) => {
                update(i, { ...lora, weight: Number(e.target.value) })
              }}
            />
            <output htmlFor={id}>{lora.weight.toFixed(2)}</output>
            <button
              type="button"
              className="lora-remove"
              aria-label={`Remove ${label}`}
              onClick={() => {
                update(i, null)
              }}
            >
              <svg viewBox="0 0 12 12" aria-hidden>
                <path d="M2 2l8 8M10 2l-8 8" />
              </svg>
            </button>
            {index && !asset && (
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
                    onClick={() => {
                      onTrigger(w)
                    }}
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
