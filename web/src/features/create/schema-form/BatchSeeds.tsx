import type { Params, SeedMode } from '@/api/types'

/**
 * The Seed row for a batch: a fixed seed would make every image the same, so it's a
 * choice between random seeds and counting up from a first seed.
 */
export function BatchSeeds({
  id,
  label,
  value,
  set,
  mode,
  onMode,
}: {
  id: string
  label: string
  value: Params[string]
  set: (value: number) => void
  mode: SeedMode
  onMode: (mode: SeedMode) => void
}) {
  const random = value === -1 || value === null
  return (
    <div className="setting seed-batch" role="group" aria-labelledby={`${id}-label`}>
      <span className="setting-label" id={`${id}-label`}>
        {label}
      </span>
      <div className="seed-batch-control">
        <div className="seed-modes">
          <button type="button" aria-pressed={mode === 'random'} onClick={() => onMode('random')}>
            Random seeds
          </button>
          <button
            type="button"
            aria-pressed={mode === 'increment'}
            onClick={() => onMode('increment')}
          >
            Count up
          </button>
        </div>
        {mode === 'increment' && (
          <div className="seed-control">
            <label htmlFor={id} className="seed-from">
              from
            </label>
            <input
              id={id}
              type="number"
              inputMode="numeric"
              placeholder="a random seed"
              value={random ? '' : String(value)}
              onChange={(e) => set(e.target.value === '' ? -1 : Number(e.target.value))}
            />
          </div>
        )}
      </div>
    </div>
  )
}
