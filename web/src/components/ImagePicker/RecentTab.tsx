import { useQuery } from '@tanstack/react-query'

import { queries } from '@/api/queries'

import { Grid } from './Grid'
import type { Picked } from './types'

export function RecentTab({ onPick }: { onPick: (p: Picked) => void }) {
  const results = useQuery(queries.results())
  if (results.isPending) return <p className="loading">Loading…</p>
  if (results.error) return <p role="alert">{results.error.message}</p>
  const items = results.data.results.map((r) => ({
    ...r,
    label: String(r.spec?.params.prompt ?? 'result'),
  }))
  return <Grid items={items} empty="No recent results. Generate something first." onPick={onPick} />
}
