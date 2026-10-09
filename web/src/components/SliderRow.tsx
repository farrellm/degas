export interface SliderRowProps {
  id: string
  label: string
  min: number
  max: number
  step?: number
  value: number
  /** How the value reads beside the track; two decimals unless given. */
  format?: (value: number) => string
  /** A line under the row saying what the setting does. */
  note?: string
  onChange: (value: number) => void
}

/** A settings row with a slider and its value. */
export function SliderRow({
  id,
  label,
  min,
  max,
  step = 0.05,
  value,
  format = (v) => v.toFixed(2),
  note,
  onChange,
}: SliderRowProps) {
  return (
    <>
      <div className="setting">
        <label className="setting-label" htmlFor={id}>
          {label}
        </label>
        <div className="slider-control">
          <input
            id={id}
            type="range"
            min={min}
            max={max}
            step={step}
            value={value}
            onChange={(e) => onChange(Number(e.target.value))}
          />
          <output htmlFor={id}>{format(value)}</output>
        </div>
      </div>
      {note && <p className="row-note slider-note">{note}</p>}
    </>
  )
}
