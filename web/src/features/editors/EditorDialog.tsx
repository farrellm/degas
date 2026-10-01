import type { ReactNode } from 'react'

import { useDialog } from '@/hooks/useDialog'

export interface EditorDialogProps {
  title: string
  /** Added to `editor`, for an editor's own layout. */
  className?: string
  /** The button that finishes the edit, opposite Cancel. */
  action?: ReactNode
  onCancel: () => void
  children: ReactNode
}

/** A full-screen editor on the image well: Cancel, the title and the finishing action in a bar. */
export function EditorDialog({ title, className, action, onCancel, children }: EditorDialogProps) {
  const ref = useDialog(onCancel)

  return (
    <div
      ref={ref}
      className={className ? `editor ${className}` : 'editor'}
      role="dialog"
      aria-modal="true"
      aria-label={title}
      tabIndex={-1}
    >
      <div className="editor-bar">
        <button type="button" className="btn quiet small" onClick={onCancel}>
          Cancel
        </button>
        <h2>{title}</h2>
        {action}
      </div>
      {children}
    </div>
  )
}
