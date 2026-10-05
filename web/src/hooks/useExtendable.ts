import { useQuery } from '@tanstack/react-query'

import { queries } from '@/api/queries'

/** Whether a family's videos can be extended from their last frame (Wan 2.2, LTX-2). */
export function useExtendable(): (family: string | undefined) => boolean {
  const families = useQuery(queries.families())
  return (family) => !!families.data?.find((f) => f.id === family)?.extendable
}
