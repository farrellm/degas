import { useState } from 'react'

import type { LoraEntry } from '@/api/types'
import { loraPaths, sameLora } from '@/lib/loras'

/** The LoRAs on the form. */
export function useLoras(initial: LoraEntry[] | undefined) {
  const [loras, setLoras] = useState<LoraEntry[]>(initial ?? [])

  return {
    loras,
    setLoras,
    /** Adding a LoRA that's already in the form does nothing. */
    add: (entry: LoraEntry) =>
      setLoras((ls) => (ls.some((l) => sameLora(l, entry)) ? ls : [...ls, entry])),
    /** Drop the LoRAs whose files were deleted from Drive. */
    drop: (paths: string[]) =>
      setLoras((ls) => ls.filter((l) => !loraPaths(l).some((p) => paths.includes(p)))),
  }
}
