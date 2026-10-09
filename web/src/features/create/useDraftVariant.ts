import { useQuery } from '@tanstack/react-query'

import { queries } from '@/api/queries'
import type { Variant } from '@/api/types'

import { loadDraft } from './draft'
import { variantOf } from './selection'

/**
 * The model variant Create has open, the way Create resolves it: the chosen model's, else
 * the first that does the chosen mode, else the family's first. Undefined until families load.
 */
export function useDraftVariant(): Variant | undefined {
  const families = useQuery(queries.families())
  const draft = loadDraft()
  const family = families.data?.find((f) => f.id === draft.family) ?? families.data?.[0]
  if (!family) return undefined
  const { model, mode } = draft.families[family.id] ?? {}
  return variantOf(family, model, mode)
}
