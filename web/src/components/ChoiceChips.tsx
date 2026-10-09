export interface ChoiceChipsProps<T extends string | number> {
  /** What is being chosen, for screen readers. */
  label: string
  /** The group's look: `mode-chips`, `segmented`, `control-traces`… */
  className: string
  options: readonly { id: T; label: string }[]
  value: T | undefined
  disabled?: boolean
  onChoose: (id: T) => void
}

/** A row of buttons of which one is pressed: a choice with every option in view. */
export function ChoiceChips<T extends string | number>({
  label,
  className,
  options,
  value,
  disabled,
  onChoose,
}: ChoiceChipsProps<T>) {
  return (
    <div className={className} role="group" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          aria-pressed={value === o.id}
          disabled={disabled}
          onClick={() => onChoose(o.id)}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}
