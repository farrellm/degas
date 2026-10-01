import type { Fit } from '@/api/types'
import { FIT_OPTIONS, type FitOption } from '@/lib/fit'

export interface FitSelectProps {
  id: string
  label?: string
  value: Fit | undefined
  options?: FitOption[]
  onChange: (fit: Fit) => void
}

/** The settings row that chooses a fit. */
export function FitSelect({
  id,
  label = 'Fit',
  value,
  options = FIT_OPTIONS,
  onChange,
}: FitSelectProps) {
  return (
    <div className="setting">
      <label className="setting-label" htmlFor={id}>
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => {
          onChange(e.target.value as Fit)
        }}
      >
        {options.map((f) => (
          <option key={f.id} value={f.id}>
            {f.label}
          </option>
        ))}
      </select>
    </div>
  )
}
