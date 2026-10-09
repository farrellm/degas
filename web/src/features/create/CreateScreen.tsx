import './create.css'
import './sources.css'

import { useState } from 'react'

import { CreateForm } from './CreateForm'
import { loadDraft } from './draft'

export interface CreateScreenProps {
  onOpenSession: () => void
  onShowResults: () => void
}

/** Create: one form per family, each keeping its own draft. */
export function CreateScreen(props: CreateScreenProps) {
  const [familyId, setFamilyId] = useState(() => loadDraft().family)
  return <CreateForm key={familyId} familyId={familyId} onFamily={setFamilyId} {...props} />
}
