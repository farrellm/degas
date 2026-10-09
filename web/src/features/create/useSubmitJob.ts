import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '@/api/client'
import { useAutoDismiss } from '@/hooks/useAutoDismiss'

import { buildJob, type FormState } from './spec'

// How long "Queued 4 images." stays up.
const QUEUED_MS = 5000

/** Queue the job a filled-in form asks for, and confirm it for a few seconds. */
export function useSubmitJob() {
  const [queued, setQueued] = useState<number | null>(null)
  useAutoDismiss(queued, QUEUED_MS, () => setQueued(null))

  const submit = useMutation({
    mutationFn: (form: FormState) => {
      const job = buildJob(form)
      return api.submitJob(job.spec, form.batchCount, job.seedMode)
    },
    onSuccess: (_, form) => setQueued(form.batchCount),
  })

  return {
    submit,
    /** How many were just queued, while the confirmation shows. */
    queued,
  }
}
