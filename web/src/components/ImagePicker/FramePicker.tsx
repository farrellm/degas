import { useMutation } from '@tanstack/react-query'
import { useRef } from 'react'

import { api } from '@/api/client'
import type { BlobInfo } from '@/api/types'
import { blobUrl } from '@/api/urls'

import type { Picked } from './types'

export function FramePicker({
  video,
  onFrame,
  onBack,
}: {
  video: Picked
  onFrame: (image: BlobInfo) => void
  onBack: () => void
}) {
  const ref = useRef<HTMLVideoElement>(null)
  const frame = useMutation({
    mutationFn: (at: 'first' | 'last' | number) => api.frame(video.sha256, at),
    onSuccess: onFrame,
  })
  return (
    <div className="picked">
      <div className="picked-image">
        <video
          ref={ref}
          src={blobUrl(video.sha256)}
          controls
          playsInline
          muted
          preload="metadata"
        />
      </div>
      <p className="picked-size">Pause on a frame, or take the first or last.</p>
      <div className="picked-actions">
        <button type="button" className="btn quiet" onClick={onBack}>
          Back
        </button>
        <button
          type="button"
          className="btn quiet"
          disabled={frame.isPending}
          onClick={() => frame.mutate('first')}
        >
          First frame
        </button>
        <button
          type="button"
          className="btn quiet"
          disabled={frame.isPending}
          onClick={() => frame.mutate('last')}
        >
          Last frame
        </button>
        <button
          type="button"
          className="btn"
          disabled={frame.isPending}
          onClick={() => frame.mutate(ref.current?.currentTime ?? 0)}
        >
          Use this frame
        </button>
      </div>
      {frame.error && <p role="alert">{frame.error.message}</p>}
    </div>
  )
}
