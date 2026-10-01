import { formatSize } from '@/lib/format'

export function AspectPicker({
  width,
  height,
  presets,
  defaultSize: [dw, dh],
  onChange,
}: {
  width: number
  height: number
  presets: [number, number][]
  /** The family's default size, notched like a slider's default. */
  defaultSize: [number, number]
  onChange: (w: number, h: number) => void
}) {
  return (
    <div className="setting stacked" role="group" aria-labelledby="aspect-label">
      <span className="setting-label" id="aspect-label">
        Size
      </span>
      <span className="setting-value">{formatSize(width, height)}</span>
      <div className="setting-control aspects">
        {presets.map(([w, h]) => {
          const on = w === width && h === height
          const fallback = w === dw && h === dh
          return (
            <button
              key={`${String(w)}x${String(h)}`}
              type="button"
              className={['aspect', w > h && 'wide', fallback && 'default']
                .filter(Boolean)
                .join(' ')}
              aria-label={`${String(w)}×${String(h)}${fallback ? ', default' : ''}`}
              aria-pressed={on}
              onClick={() => {
                onChange(w, h)
              }}
            >
              <span
                className="aspect-box"
                style={{ aspectRatio: `${String(w)} / ${String(h)}` }}
                aria-hidden
              />
            </button>
          )
        })}
      </div>
    </div>
  )
}
