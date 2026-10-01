export interface ModelRowProps {
  /** The model's name, or what to say while there is none. */
  value: string
  chosen: boolean
  /** The chosen model isn't in Drive. */
  missing: boolean
  /** The model's variant, when its name doesn't already say it. */
  variantLabel: string | null
  /** The session's GPU is below what the model needs. */
  underpowered: { needs: string; has: string } | null
  onPick: () => void
}

/** The model a job runs, and what's wrong with the choice if anything is. */
export function ModelRow({
  value,
  chosen,
  missing,
  variantLabel,
  underpowered,
  onPick,
}: ModelRowProps) {
  return (
    <div className={missing ? 'model-row missing' : 'model-row'}>
      <button
        type="button"
        className="setting setting-button"
        aria-describedby={missing ? 'model-missing' : undefined}
        onClick={onPick}
      >
        <span className="setting-label">Model</span>{' '}
        <span className={chosen ? 'setting-value' : 'setting-value none'}>{value}</span>
      </button>
      {missing && (
        <p className="row-warning" id="model-missing">
          Not found in Drive. Pick another model.
        </p>
      )}
      {variantLabel && <p className="row-note">{variantLabel}</p>}
      {underpowered && (
        <p className="row-warning">
          Needs an {underpowered.needs}; this {underpowered.has} session may run it slowly.
        </p>
      )}
    </div>
  )
}
