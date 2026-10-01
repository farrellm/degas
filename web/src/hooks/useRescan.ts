import { useMutation, useQueryClient } from '@tanstack/react-query'

import { api } from '@/api/client'
import { queryKeys } from '@/api/queries'

/** Index Drive again, then refresh what was read from the old index. */
export function useRescan() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: api.rescan,
    onSettled: () => qc.invalidateQueries({ queryKey: queryKeys.drive }),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.assets }),
  })
}
