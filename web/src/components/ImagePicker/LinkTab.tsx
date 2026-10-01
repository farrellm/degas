import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '@/api/client'

import type { Picked } from './types'

export function LinkTab({ onPick }: { onPick: (p: Picked) => void }) {
  const [url, setUrl] = useState('')
  const [pasteFailed, setPasteFailed] = useState(false)
  const fetchUrl = useMutation({ mutationFn: api.fromUrl, onSuccess: onPick })
  const canPaste = typeof navigator !== 'undefined' && 'clipboard' in navigator
  return (
    <form
      className="picker-source"
      onSubmit={(e) => {
        e.preventDefault()
        if (url.trim()) fetchUrl.mutate(url.trim())
      }}
    >
      <div className="link-row">
        <input
          type="url"
          inputMode="url"
          aria-label="Image link"
          placeholder="https://…"
          autoCapitalize="none"
          autoCorrect="off"
          value={url}
          onChange={(e) => {
            setUrl(e.target.value)
          }}
        />
        {canPaste && (
          <button
            type="button"
            className="btn quiet"
            onClick={() => {
              setPasteFailed(false)
              navigator.clipboard
                .readText()
                .then((text) => {
                  setUrl(text.trim())
                })
                .catch(() => {
                  setPasteFailed(true)
                })
            }}
          >
            Paste
          </button>
        )}
      </div>
      <button type="submit" className="btn" disabled={!url.trim() || fetchUrl.isPending}>
        {fetchUrl.isPending ? 'Importing…' : 'Import'}
      </button>
      <p>A link to an image, or directly to an MP4 or WebM video.</p>
      {pasteFailed && <p role="alert">Couldn’t read the clipboard. Paste into the field.</p>}
      {fetchUrl.error && <p role="alert">{fetchUrl.error.message}</p>}
    </form>
  )
}
