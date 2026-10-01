import { useQuery } from '@tanstack/react-query'

import { queries } from '@/api/queries'
import { useAssets } from '@/hooks/useAssets'
import { isActive, isGpuReady } from '@/lib/session'

export const SAM_ASSET = 'preprocessors/sam3'

/** Why Select can't be used right now (SAM 3 isn't in Drive, or no GPU is ready); null when it can. */
export function useSelectNote(): string | null {
  const session = useQuery(queries.session())
  const assets = useAssets()
  const samIndexed = assets.data?.some((a) => a.path === SAM_ASSET) ?? false
  if (!samIndexed) return `Put SAM 3 in Drive under degas/${SAM_ASSET}/ and rescan to use Select.`
  if (isGpuReady(session.data)) return null
  return isActive(session.data)
    ? 'Select works once the GPU session is ready.'
    : 'Start a session to use Select.'
}
