import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { useDialog } from './useDialog'

function Dialog({ onEscape }: { onEscape: () => void }) {
  const ref = useDialog(onEscape)
  return (
    <div ref={ref} role="dialog" tabIndex={-1}>
      <input aria-label="Search" />
    </div>
  )
}

describe('useDialog', () => {
  it('keeps focus where it is when the parent re-renders with a new onEscape', async () => {
    const user = userEvent.setup()
    const { rerender } = render(<Dialog onEscape={() => undefined} />)
    expect(screen.getByRole('dialog')).toHaveFocus()
    await user.click(screen.getByRole('textbox', { name: 'Search' }))
    rerender(<Dialog onEscape={() => undefined} />)
    expect(screen.getByRole('textbox', { name: 'Search' })).toHaveFocus()
  })

  it('calls the latest onEscape', async () => {
    const user = userEvent.setup()
    const first = vi.fn()
    const latest = vi.fn()
    const { rerender } = render(<Dialog onEscape={first} />)
    rerender(<Dialog onEscape={latest} />)
    await user.keyboard('{Escape}')
    expect(first).not.toHaveBeenCalled()
    expect(latest).toHaveBeenCalledOnce()
  })
})
