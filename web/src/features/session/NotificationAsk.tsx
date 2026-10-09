import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import type { Job } from '@/api/types'

import { enablePush, isInstalled, markAsked, pushQuery, pushSupported, wasAsked } from './push'

/**
 * Asked once, after the first job finishes while Degas is installed to the home screen.
 * After that, notifications are a switch in the GPU session sheet.
 */
export function NotificationAsk({ jobs }: { jobs: Job[] | undefined }) {
  const qc = useQueryClient()
  const [dismissed, setDismissed] = useState(false)
  const allow = useMutation({
    mutationFn: enablePush,
    onSuccess: (state) => {
      qc.setQueryData(pushQuery().queryKey, state)
      setDismissed(true)
    },
  })
  const due =
    !dismissed &&
    !!jobs?.some((j) => j.status === 'done') &&
    pushSupported() &&
    isInstalled() &&
    Notification.permission === 'default' &&
    !wasAsked()
  if (!due) return null

  return (
    <div className="ask" role="dialog" aria-labelledby="ask-question">
      <p id="ask-question">Get a notification when images finish?</p>
      <div className="ask-actions">
        <button
          type="button"
          className="btn quiet small"
          onClick={() => {
            markAsked()
            setDismissed(true)
          }}
        >
          Not now
        </button>
        <button
          type="button"
          className="btn small"
          disabled={allow.isPending}
          onClick={() => allow.mutate()}
        >
          Allow
        </button>
      </div>
      {allow.error && <p role="alert">{allow.error.message}</p>}
    </div>
  )
}
