import { useState } from 'react'

import type { BlobInfo } from '@/api/types'
import { blobUrl } from '@/api/urls'
import { Sheet } from '@/components/Sheet'
import { formatSize } from '@/lib/format'
import { isVideo } from '@/lib/image'

import { FramePicker } from './FramePicker'
import { LibraryTab } from './LibraryTab'
import { LinkTab } from './LinkTab'
import { PhotosTab } from './PhotosTab'
import { RecentTab } from './RecentTab'
import type { Picked } from './types'

const TABS = ['Recent', 'Library', 'Photos', 'Link'] as const
type Tab = (typeof TABS)[number]

export interface ImagePickerProps {
  onUse: (image: BlobInfo) => void
  /** Offers Crop when given. */
  onCrop?: (image: BlobInfo) => void
  onClose: () => void
}

/**
 * Choose a source image from recent results, the library, the camera roll or a link.
 * A video offers a frame to use (design §8.2).
 */
export function ImagePicker({ onUse, onCrop, onClose }: ImagePickerProps) {
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
            {tab === 'Recent' && <RecentTab onPick={setPicked} />}
            {tab === 'Library' && <LibraryTab onPick={setPicked} />}
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
            {picked.width && picked.height ? formatSize(picked.width, picked.height) : null}
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
            {onCrop && (
              <button
                type="button"
                className="btn quiet"
                onClick={() => {
                  onCrop(picked)
                }}
              >
                Crop
              </button>
            )}
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
