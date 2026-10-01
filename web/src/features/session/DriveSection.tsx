import { useQuery } from '@tanstack/react-query'

import { queries } from '@/api/queries'
import type { SessionSnapshot } from '@/api/types'
import { useNow } from '@/hooks/useNow'
import { useRescan } from '@/hooks/useRescan'
import { ago } from '@/lib/time'

export function DriveSection({ authorizedHint }: { authorizedHint?: SessionSnapshot['drive'] }) {
  const now = useNow(60_000)
  const drive = useQuery(queries.drive())
  const rescan = useRescan()
  const d = drive.data
  const problem = d?.error ?? authorizedHint?.push_error

  return (
    <section className="sheet-section" aria-label="Google Drive">
      <h3>Google Drive</h3>
      {!d ? null : !d.configured ? (
        <p>
          No OAuth client is set. Add <code>drive.client_file</code> to <code>degas.toml</code>.
        </p>
      ) : !d.authorized ? (
        <p>
          Drive isn't authorized. Run <code>degas auth drive</code> on the server.
        </p>
      ) : (
        <p>
          Models indexed {d.indexed_at ? ago(d.indexed_at, now) : 'never'}.
          {rescan.data && ` Found ${String(rescan.data.count)} files.`}
        </p>
      )}
      {problem && <p className="problem">{problem}</p>}
      <div className="sheet-actions">
        <button
          type="button"
          className="btn quiet"
          disabled={rescan.isPending || !d?.authorized}
          onClick={() => {
            rescan.mutate()
          }}
        >
          {rescan.isPending ? 'Rescanning…' : 'Rescan Drive'}
        </button>
      </div>
      {rescan.error && <p role="alert">{rescan.error.message}</p>}
    </section>
  )
}
