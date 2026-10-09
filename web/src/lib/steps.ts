// The part of a run a ControlNet unit or an image prompt acts on, kept as fractions of the
// steps (`start`, `end` in 0..1) so it survives a change of step count.

/**
 * The steps (1-based, inclusive) a unit guides out of `steps`. diffusers runs a unit on step
 * i (0-based) when i / n ≥ start and (i + 1) / n ≤ end; null when that's no step at all.
 */
export function stepSpan(
  start: number,
  end: number,
  steps: number,
): { first: number; last: number } | null {
  const first = Math.ceil(start * steps - 1e-9) + 1
  const last = Math.floor(end * steps + 1e-9)
  return steps > 0 && last >= first ? { first, last } : null
}

/** "Steps 1–24 of 30". */
export function stepsLabel(start: number, end: number, steps: number): string {
  const span = stepSpan(start, end, steps)
  if (!span) return 'No steps. Widen the range.'
  if (span.first === span.last) return `Step ${span.first} of ${steps}`
  return `Steps ${span.first}–${span.last} of ${steps}`
}

/** A step as a fraction of the run, to the three places a job spec carries. */
export const stepFraction = (step: number, steps: number) =>
  Math.round((step / steps) * 1000) / 1000
