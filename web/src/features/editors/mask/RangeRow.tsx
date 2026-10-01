export interface RangeRowProps {
  id: string
  label: string
  min: number
  max: number
  step?: number
  value: number
  /** The value as it reads beside the track: "24 px", "80%". */
  output: string
  onChange: (value: number) => void
}

/** A slider in the mask editor's controls. */
export function RangeRow({ id, label, min, max, step, value, output, onChange }: RangeRowProps) {
  return (
    <div className="mask-row">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          onChange(Number(e.target.value))
        }}
      />
      <output htmlFor={id}>{output}</output>
    </div>
  )
}
