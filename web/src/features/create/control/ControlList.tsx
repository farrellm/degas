import './control.css'

import type { Asset } from '@/api/types'
import { assetLabel } from '@/lib/assets'

import { type ControlUnit, MAX_UNITS, unitSummary } from './control'
import { ControlThumb } from './ControlThumb'

export interface ControlListProps {
  units: ControlUnit[]
  /** The family's ControlNets in the Drive index. */
  index: Asset[] | undefined
  steps: number
  onOpen: (key: string) => void
  onAdd: () => void
}

/** The ControlNet rows in Create: each unit's image, model and summary; tap to edit. */
export function ControlList({ units, index, steps, onOpen, onAdd }: ControlListProps) {
  return (
    <div className="control-group" role="group" aria-labelledby="control-label">
      <div className="setting">
        <span className="setting-label" id="control-label">
          ControlNet
        </span>
        <button
          type="button"
          className="row-action"
          aria-label="Add ControlNet"
          disabled={units.length >= MAX_UNITS}
          onClick={onAdd}
        >
          Add
        </button>
      </div>
      {units.map((unit) => {
        const missing = !!unit.model && !!index && !index.some((a) => a.path === unit.model)
        const name = unit.model ? assetLabel(unit.model, index) : 'Choose a ControlNet'
        return (
          <div key={unit.key} className={missing ? 'control-unit missing' : 'control-unit'}>
            <button
              type="button"
              className="control-open"
              aria-label={`Edit ControlNet: ${name}`}
              onClick={() => onOpen(unit.key)}
            >
              {unit.image ? (
                <ControlThumb image={unit.image.sha} area={unit.area} />
              ) : (
                <span className="control-thumb empty" aria-hidden />
              )}
              <span className="control-text">
                <span className={unit.model ? 'control-name' : 'control-name none'}>{name}</span>
                <span className="control-meta">{unitSummary(unit, steps)}</span>
              </span>
            </button>
            {missing && <p className="row-warning">Not found in Drive. Pick another ControlNet.</p>}
          </div>
        )
      })}
    </div>
  )
}
