import { useQuery } from '@tanstack/react-query'

import { queries } from '@/api/queries'

import { loadDraft } from './draft'
import { SOURCE_MODES } from './modes'

/**
 * Where "Use as source" sends an image: the family Create has open (image-to-image for
 * pictures, image-to-video for clips), staying in its current image mode if it has one.
 * Families that can't start from an image are skipped.
 */
export function useSourceTarget(): { family: string; mode: string } | null {
  const families = useQuery(queries.families())
  const draft = loadDraft()
  const list = families.data ?? []
  const ordered = [
    ...list.filter((f) => f.id === draft.family),
    ...list.filter((f) => f.id !== draft.family),
  ]
  for (const f of ordered) {
    const modes = [...new Set(f.variants.flatMap((v) => v.modes))].filter((m) =>
      SOURCE_MODES.has(m),
    )
    const current = draft.families[f.id]?.mode
    const mode = current && modes.includes(current) ? current : modes[0]
    if (mode) return { family: f.id, mode }
  }
  return null
}
