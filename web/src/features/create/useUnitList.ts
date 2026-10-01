import { useState } from 'react'

/**
 * A list of units (ControlNets, image prompts) of which one at a time is open in its editor.
 * Units are known by their `key`.
 */
export function useUnitList<T extends { key: string }>(initial: T[]) {
  const [units, setUnits] = useState(initial)
  const [editing, setEditing] = useState<string | null>(null)

  return {
    units,
    /** The unit open in the editor, if it's still in the list. */
    current: units.find((u) => u.key === editing),
    open: setEditing,
    /** Add a unit and open it. */
    add: (unit: T) => {
      setUnits([...units, unit])
      setEditing(unit.key)
    },
    /** Change the open unit. */
    update: (change: (unit: T) => T) => {
      setUnits((us) => us.map((u) => (u.key === editing ? change(u) : u)))
    },
    /** Remove the open unit. */
    remove: () => {
      setUnits((us) => us.filter((u) => u.key !== editing))
      setEditing(null)
    },
    /** Close the editor; a unit left not `worthKeeping` goes with it. */
    close: (worthKeeping: (unit: T) => boolean) => {
      setUnits((us) => us.filter((u) => u.key !== editing || worthKeeping(u)))
      setEditing(null)
    },
  }
}
