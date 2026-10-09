import type { SessionSnapshot } from '@/api/types'
import { useAssets } from '@/hooks/useAssets'
import { assetLabel } from '@/lib/assets'
import { formatBytes } from '@/lib/format'

type Cache = NonNullable<NonNullable<SessionSnapshot['worker']>['cache']>

/** Models and LoRAs already copied to the VM, so choosing them costs no copy. */
export function CacheSection({ cache }: { cache: Cache }) {
  const assets = useAssets()
  return (
    <section className="sheet-section" aria-labelledby="cache-heading">
      <h3 id="cache-heading">On the GPU</h3>
      {cache.files.length === 0 ? (
        <p>Nothing copied yet. Models and LoRAs copy from Drive the first time a job uses them.</p>
      ) : (
        <>
          <div>
            <div className="meter" aria-hidden>
              <span style={{ width: `${Math.min(100, (100 * cache.used) / cache.budget)}%` }} />
            </div>
            <p className="bar-note">
              {formatBytes(cache.used)} of {formatBytes(cache.budget)} used. The least recently used
              files are removed above that.
            </p>
          </div>
          <ul className="cached">
            {cache.files.map((f) => (
              <li key={f.path}>
                <span>{assetLabel(f.path, assets.data)}</span>
                <span className="size">{formatBytes(f.size)}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}
