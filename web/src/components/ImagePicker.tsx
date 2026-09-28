import { useMutation, useQuery } from '@tanstack/react-query'
import { useRef, useState, type CSSProperties } from 'react'
import { api, blobUrl, isVideo, thumbUrl, type BlobInfo } from '../api'
import { clock, size } from '../format'
import { Sheet } from './Sheet'

const TABS = ['Recent', 'Library', 'Photos', 'Link'] as const
type Tab = (typeof TABS)[number]

/** A picked image or video, before it goes into the slot. */
type Picked = BlobInfo

interface Props {
  onUse: (image: BlobInfo) => void
  onCrop: (image: BlobInfo) => void
  onClose: () => void
}

/**
 * Choose a source image from recent results, the library, the camera roll or a link.
 * A video offers a frame to use (design §8.2).
 */
export function ImagePicker({ onUse, onCrop, onClose }: Props) {
  const [tab, setTab] = useState<Tab>('Recent')
  const [picked, setPicked] = useState<Picked | null>(null)

  return (
    <Sheet title="Choose image" onClose={onClose}>
      {picked === null ? (
        <>
          <div className="segmented tabs-4" role="tablist" aria-label="Where from">
            {TABS.map((t) => (
              <button
                key={t}
                type="button"
                role="tab"
                aria-selected={t === tab}
                onClick={() => {
                  setTab(t)
                }}
              >
                {t}
              </button>
            ))}
          </div>
          <div className="picker-body" role="tabpanel" aria-label={tab}>
            {tab === 'Recent' && <RecentGrid onPick={setPicked} />}
            {tab === 'Library' && <LibraryGrid onPick={setPicked} />}
            {tab === 'Photos' && <PhotosTab onPick={setPicked} />}
            {tab === 'Link' && <LinkTab onPick={setPicked} />}
          </div>
        </>
      ) : isVideo(picked.media_type) ? (
        <FramePicker
          video={picked}
          onFrame={setPicked}
          onBack={() => {
            setPicked(null)
          }}
        />
      ) : (
        <div className="picked">
          <div className="picked-image">
            <img src={blobUrl(picked.sha256)} alt="The chosen image" />
          </div>
          <p className="picked-size">
            {picked.width && picked.height ? size(picked.width, picked.height) : null}
          </p>
          <div className="picked-actions">
            <button
              type="button"
              className="btn quiet"
              onClick={() => {
                setPicked(null)
              }}
            >
              Back
            </button>
            <button
              type="button"
              className="btn quiet"
              onClick={() => {
                onCrop(picked)
              }}
            >
              Crop
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => {
                onUse(picked)
              }}
            >
              Use image
            </button>
          </div>
        </div>
      )}
    </Sheet>
  )
}

interface GridItem {
  id: string
  blob_sha: string
  media_type: string
  width: number | null
  height: number | null
  duration: number | null
  label: string
}

function Grid({
  items,
  empty,
  onPick,
}: {
  items: GridItem[]
  empty: string
  onPick: (p: Picked) => void
}) {
  if (items.length === 0) return <p className="asset-empty">{empty}</p>
  return (
    <div className="picker-grid">
      {items.map((it) => (
        <button
          key={it.id}
          type="button"
          className="tile"
          style={
            { '--ratio': `${String(it.width ?? 1)} / ${String(it.height ?? 1)}` } as CSSProperties
          }
          aria-label={`${isVideo(it.media_type) ? 'Video' : 'Image'}: ${it.label}`}
          onClick={() => {
            onPick({
              sha256: it.blob_sha,
              media_type: it.media_type,
              width: it.width,
              height: it.height,
              duration: it.duration,
            })
          }}
        >
          <img src={thumbUrl(it.blob_sha)} alt="" loading="lazy" />
          {isVideo(it.media_type) && it.duration !== null && (
            <span className="tile-duration">{clock(it.duration)}</span>
          )}
        </button>
      ))}
    </div>
  )
}

function RecentGrid({ onPick }: { onPick: (p: Picked) => void }) {
  const results = useQuery({ queryKey: ['results'], queryFn: () => api.results() })
  if (results.isPending) return <p className="loading">Loading…</p>
  if (results.error) return <p role="alert">{results.error.message}</p>
  const items = results.data.results.map((r) => ({
    ...r,
    label: String(r.spec?.params.prompt ?? 'result'),
  }))
  return <Grid items={items} empty="No recent results. Generate something first." onPick={onPick} />
}

function LibraryGrid({ onPick }: { onPick: (p: Picked) => void }) {
  const library = useQuery({ queryKey: ['library', 'picker'], queryFn: () => api.library('') })
  if (library.isPending) return <p className="loading">Loading…</p>
  if (library.error) return <p role="alert">{library.error.message}</p>
  const items = library.data.items.map((i) => ({
    ...i,
    label: String(i.config.params.prompt ?? 'kept item'),
  }))
  return <Grid items={items} empty="Nothing kept yet." onPick={onPick} />
}

function PhotosTab({ onPick }: { onPick: (p: Picked) => void }) {
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

function LinkTab({ onPick }: { onPick: (p: Picked) => void }) {
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

function FramePicker({
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
          onClick={() => {
            frame.mutate('first')
          }}
        >
          First frame
        </button>
        <button
          type="button"
          className="btn quiet"
          disabled={frame.isPending}
          onClick={() => {
            frame.mutate('last')
          }}
        >
          Last frame
        </button>
        <button
          type="button"
          className="btn"
          disabled={frame.isPending}
          onClick={() => {
            frame.mutate(ref.current?.currentTime ?? 0)
          }}
        >
          Use this frame
        </button>
      </div>
      {frame.error && <p role="alert">{frame.error.message}</p>}
    </div>
  )
}
