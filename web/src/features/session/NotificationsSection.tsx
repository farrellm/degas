import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { disablePush, enablePush, isInstalled, type PushState, pushState } from './push'

/** Web Push for this device: finished jobs and the idle warning. */
export function NotificationsSection() {
  const qc = useQueryClient()
  const state = useQuery({ queryKey: ['push'], queryFn: pushState })
  const toggle = useMutation({
    mutationFn: (on: boolean): Promise<PushState> => (on ? enablePush() : disablePush()),
    onSuccess: (next) => {
      qc.setQueryData(['push'], next)
    },
  })
  const s = state.data
  if (!s) return null

  return (
    <section className="sheet-section" aria-labelledby="notify-heading">
      <h3 id="notify-heading">Notifications</h3>
      {s === 'unsupported' ? (
        <p>
          {isInstalled()
            ? 'This device can’t show notifications from Degas.'
            : 'To get notifications, add Degas to the Home Screen: tap Share, then Add to Home Screen.'}
        </p>
      ) : s === 'blocked' ? (
        <p>Notifications for Degas are turned off in Settings.</p>
      ) : (
        <label className="switch">
          <span>
            When images finish
            <small>And 2 minutes before an idle session stops.</small>
          </span>
          <input
            type="checkbox"
            checked={s === 'on'}
            disabled={toggle.isPending}
            onChange={(e) => {
              toggle.mutate(e.target.checked)
            }}
          />
        </label>
      )}
      {toggle.error && <p role="alert">{toggle.error.message}</p>}
    </section>
  )
}
