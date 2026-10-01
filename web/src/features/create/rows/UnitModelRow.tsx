export interface UnitModelRowProps {
  /** The chosen model's name, or null when none is chosen. */
  value: string | null
  placeholder: string
  /** Said when the chosen model isn't in Drive; null when it is. */
  missing: string | null
  /** Why the model doesn't suit the unit, if it doesn't. */
  warning: string | null | undefined | false
  onPick: () => void
}

/** The model a ControlNet unit or an image prompt uses. */
export function UnitModelRow({ value, placeholder, missing, warning, onPick }: UnitModelRowProps) {
  return (
    <div className={missing ? 'model-row missing' : 'model-row'}>
      <button type="button" className="setting setting-button" onClick={onPick}>
        <span className="setting-label">Model</span>{' '}
        <span className={value ? 'setting-value' : 'setting-value none'}>
          {value ?? placeholder}
        </span>
      </button>
      {missing && <p className="row-warning">{missing}</p>}
      {warning && !missing && <p className="row-warning">{warning}</p>}
    </div>
  )
}
