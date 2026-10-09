import { useEffect, useEffectEvent } from 'react'

import type { SeedMode } from '@/api/types'

import { type FamilyDraft, saveFamilyDraft } from './draft'

/**
 * Keep `family`'s draft saved as the form changes (`draft` is null while the form loads).
 * Saved when something in it changed, not at every render: `draft` is built afresh each time.
 */
export function useSaveDraft(
  family: string,
  draft: FamilyDraft | null,
  batchCount: number,
  seedMode: SeedMode,
) {
  const changed = JSON.stringify([family, draft, batchCount, seedMode])
  const save = useEffectEvent(() => {
    if (draft) saveFamilyDraft(family, draft, batchCount, seedMode)
  })
  useEffect(() => save(), [changed])
}
