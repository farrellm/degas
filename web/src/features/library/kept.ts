import type { LibraryItem } from '@/api/types'
import type { ViewerItem } from '@/components/Viewer/Viewer'

export type Kept = LibraryItem & ViewerItem

export function asViewerItem(item: LibraryItem): Kept {
  const seed = item.config.params.seed
  return {
    ...item,
    seed: typeof seed === 'number' ? seed : null,
    spec: item.config,
    segments: item.config.segments ?? null,
  }
}
