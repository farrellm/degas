import { useSyncExternalStore } from 'react'

import { type Draft, loadDraft, subscribeDraft } from './draft'

/** The Create draft, re-rendering when it's saved (here or in another tab). */
export function useDraft(): Draft {
  return useSyncExternalStore(subscribeDraft, loadDraft)
}
