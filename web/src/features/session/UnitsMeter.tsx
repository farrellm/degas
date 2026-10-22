import './UnitsMeter.css'

import type { SessionSnapshot } from '@/api/types'
import { balanceNow, duration, formatUnits, hoursLeft, sessionSpend } from '@/lib/units'

const LOW_HOURS = 10 // the balance line turns ochre below this many hours at the current rate

/** What the running session has spent, counting up, and the account balance under it. */
export function UnitsMeter({ snap, now }: { snap: SessionSnapshot; now: number }) {
  const spend = sessionSpend(snap.session, now)
  const rate = snap.session?.unit_rate
  return (
    <>
      {spend !== null && rate != null && (
        <div className="units-spent">
          <p className="figure">
            <Readout value={spend} /> units
          </p>
          <p>spent this session, at {formatUnits(rate)} an hour</p>
        </div>
      )}
      <UnitsLeft snap={snap} now={now} />
    </>
  )
}

/** Tabular digits, so they hold still as they tick, around a proportional point: this face's
 * tabular point is as wide as a digit. */
function Readout({ value }: { value: number }) {
  const [whole, fraction] = formatUnits(value).split('.')
  return (
    <span className="readout">
      {whole}
      <span className="point">.</span>
      {fraction}
    </span>
  )
}

/** The account balance and how long it lasts at the current rate. */
function UnitsLeft({ snap, now }: { snap: SessionSnapshot; now: number }) {
  if (!snap.units) return null
  const balance = balanceNow(snap.units, now)
  const hours = hoursLeft(balance, snap.units.rate)
  return (
    <p className={`units-left${hours !== null && hours < LOW_HOURS ? ' warn' : ''}`}>
      {formatUnits(balance)} compute units left
      {hours !== null && `, about ${duration(hours)} at this rate`}.
    </p>
  )
}

/** Under the start button: the balance, and what the last session used. */
export function UnitsSummary({ snap, now }: { snap: SessionSnapshot; now: number }) {
  const last = snap.session?.ended_at ? sessionSpend(snap.session, now) : null
  if (!snap.units && last === null) return null
  return (
    <p className="units-left">
      {snap.units && `${formatUnits(balanceNow(snap.units, now))} compute units left. `}
      {last !== null &&
        `The last session used ${formatUnits(last)}${snap.units ? '' : ' compute units'}.`}
    </p>
  )
}
