import './Sheet.css'

import type { ReactNode } from 'react'

import { useDialog } from '@/hooks/useDialog'

export interface SheetProps {
  title: string
  /** Beside Done in the header (the LoRA picker's Delete LoRAs). */
  actions?: ReactNode
  onClose: () => void
  children: ReactNode
}

/** Modal bottom sheet: Escape or the scrim closes it, focus returns to the opener. */
export function Sheet({ title, actions, onClose, children }: SheetProps) {
  const ref = useDialog(onClose)

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
