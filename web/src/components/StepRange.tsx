import { stepsLabel, stepSpan } from '@/lib/steps'

export interface StepRangeProps {
  /** The form's step count. */
  steps: number
  /** The first and last step of the range, counted in `steps`. */
  a: number
  b: number
  onChange: (a: number, b: number) => void
  /** What the range is for, under the track. */
  note?: string
}

/**
 * The steps a unit guides, as two thumbs on one track counted in the form's steps. Early
 * steps set the layout, so ending early leaves the details free.
 */
export function StepRange({
  steps,
  a,
  b,
  onChange,
  note = 'Early steps set the layout; ending early leaves the details free.',
}: StepRangeProps) {
  const n = Math.max(1, steps)
  const pct = (x: number) => `${(x / n) * 100}%`
  const span = stepSpan(a / n, b / n, n)
  return (
    <div className="setting stacked step-range" role="group" aria-labelledby="step-range-label">
      <span className="setting-label" id="step-range-label">
        Steps
      </span>
      <output className="readout" aria-live="polite">
        {stepsLabel(a / n, b / n, n).replace(/^Steps? /, '')}
      </output>
      <div className="setting-control">
        <div className="step-track">
          <span
            className={span ? 'step-span' : 'step-span none'}
            style={{ left: pct(a), right: `calc(100% - ${pct(b)})` }}
          />
          <input
            type="range"
            aria-label="First step"
            min={0}
            max={n}
            step={1}
            value={a}
            onChange={(e) => onChange(Math.min(Number(e.target.value), b - 1), b)}
          />
          <input
            type="range"
            aria-label="Last step"
            min={0}
            max={n}
            step={1}
            value={b}
            onChange={(e) => onChange(a, Math.max(Number(e.target.value), a + 1))}
          />
        </div>
        <p className="row-note">{note}</p>
      </div>
    </div>
  )
}
