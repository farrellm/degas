import { useQuery } from '@tanstack/react-query'

import { queries } from '@/api/queries'
import { isActive } from '@/lib/session'

/** The whole Drive index, fetched once and filtered on the phone. */
export function useAssets() {
  return useQuery(queries.assets())
}

/** Paths copied to the running session's GPU, or null when there's no session to ask. */
export function useCachedPaths(): Set<string> | null {
  const session = useQuery(queries.session())
  const cache = session.data?.worker?.cache
  if (!isActive(session.data) || !cache) return null
  return new Set(cache.files.map((f) => f.path))
}
