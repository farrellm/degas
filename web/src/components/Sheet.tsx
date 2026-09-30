import { useEffect, useRef, type ReactNode } from 'react'

interface Props {
  title: string
  /** Beside Done in the header (the LoRA picker's Delete LoRAs). */
  actions?: ReactNode
  onClose: () => void
  children: ReactNode
}

/** Modal bottom sheet: Escape or the scrim closes it, focus returns to the opener. */
export function Sheet({ title, actions, onClose, children }: Props) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null
    ref.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = overflow
      opener?.focus()
    }
  }, [onClose])

  return (
    <>
      <div className="scrim" onClick={onClose} aria-hidden />
      <div
        ref={ref}
        className="sheet"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
      >
        <div className="sheet-inner">
          <div className="sheet-head">
            <h2>{title}</h2>
            <div className="sheet-actions">
              {actions}
              <button type="button" className="btn quiet small" onClick={onClose}>
                Done
              </button>
            </div>
          </div>
          {children}
        </div>
      </div>
    </>
  )
}
