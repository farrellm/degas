import { useMutation } from '@tanstack/react-query'
import { useRef } from 'react'

import { api } from '@/api/client'

import type { Picked } from './types'

export function PhotosTab({ onPick }: { onPick: (p: Picked) => void }) {
  const input = useRef<HTMLInputElement>(null)
  const upload = useMutation({ mutationFn: api.upload, onSuccess: onPick })
  return (
    <div className="picker-source">
      <input
        ref={input}
        type="file"
        accept="image/*,video/*"
        className="visually-hidden"
        aria-label="Photo or video"
        tabIndex={-1}
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) upload.mutate(file)
          e.target.value = ''
        }}
      />
      <button
        type="button"
        className="btn wide-btn"
        disabled={upload.isPending}
        onClick={() => input.current?.click()}
      >
        {upload.isPending ? 'Uploading…' : 'Choose from Photos'}
      </button>
      <p>A photo or a video; for a video you choose the frame next.</p>
      {upload.error && <p role="alert">{upload.error.message}</p>}
    </div>
  )
}
