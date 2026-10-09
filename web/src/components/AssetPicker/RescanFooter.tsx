import { useQuery } from '@tanstack/react-query'

import { queries } from '@/api/queries'
import { useNow } from '@/hooks/useNow'
import { useRescan } from '@/hooks/useRescan'
import { ago } from '@/lib/time'

export function RescanFooter() {
  const now = useNow(60_000)
  const drive = useQuery(queries.drive())
  const rescan = useRescan()
  const d = drive.data
  return (
    <div className="asset-footer">
      <p>
        {!d
          ? null
          : !d.authorized
            ? 'Drive isn’t authorized yet.'
            : `Indexed ${d.indexed_at ? ago(d.indexed_at, now) : 'never'}.`}
      </p>
      <button
        type="button"
        className="btn quiet small"
        disabled={rescan.isPending || !d?.authorized}
        onClick={() => rescan.mutate()}
      >
        {rescan.isPending ? 'Rescanning…' : 'Rescan Drive'}
      </button>
      {rescan.error && <p role="alert">{rescan.error.message}</p>}
    </div>
  )
}
