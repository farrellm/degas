import { useEffect, useEffectEvent, useRef } from 'react'

/**
 * What every modal surface does while it's open: take focus, stop the page behind from
 * scrolling, close on Escape (when `onEscape` is given), and hand focus back to whatever
 * opened it. Put the returned ref on the dialog element, which needs `tabIndex={-1}`.
 */
export function useDialog<T extends HTMLElement = HTMLDivElement>(onEscape?: () => void) {
  const ref = useRef<T>(null)
  // Callers pass a fresh closure each render; re-running the effect would move focus.
  const escape = useEffectEvent(() => {
    onEscape?.()
  })

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    ref.current?.focus()
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') escape()
    }
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = overflow
      opener?.focus()
    }
  }, [])

  return ref
}
