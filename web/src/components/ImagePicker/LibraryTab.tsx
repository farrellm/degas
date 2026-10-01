import { useQuery } from '@tanstack/react-query'

import { queries } from '@/api/queries'

import { Grid } from './Grid'
import type { Picked } from './types'

export function LibraryTab({ onPick }: { onPick: (p: Picked) => void }) {
  const library = useQuery(queries.libraryPicker())
  if (library.isPending) return <p className="loading">Loading…</p>
  if (library.error) return <p role="alert">{library.error.message}</p>
  const items = library.data.items.map((i) => ({
    ...i,
    label: String(i.config.params.prompt ?? 'kept item'),
  }))
  return <Grid items={items} empty="Nothing kept yet." onPick={onPick} />
}
