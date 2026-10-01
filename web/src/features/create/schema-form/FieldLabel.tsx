import type { ParamProp, Params } from '@/api/types'
import { resetLabel } from '@/lib/schema'

/**
 * A setting's label, with "Reset to 28" under it once the value has moved off the family's
 * default. It sits in the label column so the row doesn't grow while a slider is dragged.
 */
export function FieldLabel({
  id,
  label,
  prop,
  value,
  reset,
}: {
  id: string
  label: string
  prop: ParamProp
  value: Params[string]
  reset: () => void
}) {
  const to = resetLabel(prop, value)
  return (
    <div className="setting-label">
      <label htmlFor={id}>{label}</label>
      {to !== null && (
        <button
          type="button"
          className="setting-reset"
          aria-label={`Reset ${label} to ${to}`}
          onClick={reset}
        >
          Reset to {to}
        </button>
      )}
    </div>
  )
}
