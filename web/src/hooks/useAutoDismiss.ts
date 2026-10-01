import { useEffect } from 'react'

/** Clear a transient message (a toast) `ms` after it was set; `value` is null while there's none. */
export function useAutoDismiss(value: unknown, ms: number, dismiss: () => void) {
  useEffect(() => {
    if (value === null) return
    const id = setTimeout(dismiss, ms)
    return () => {
      clearTimeout(id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `dismiss` is a fresh closure each render
  }, [value, ms])
}
