import { useEffect, useEffectEvent } from 'react'

/** Clear a transient message (a toast) `ms` after it was set; `value` is null while there's none. */
export function useAutoDismiss(value: unknown, ms: number, dismiss: () => void) {
  const onDismiss = useEffectEvent(dismiss)

  useEffect(() => {
    if (value === null) return
    const id = setTimeout(() => onDismiss(), ms)
    return () => clearTimeout(id)
  }, [value, ms])
}
