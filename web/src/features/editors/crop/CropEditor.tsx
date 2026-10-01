import { useMutation, useQuery } from '@tanstack/react-query'
import { type CSSProperties, useRef } from 'react'

import { api } from '@/api/client'
import { queries } from '@/api/queries'
import type { BlobInfo } from '@/api/types'
import { blobUrl } from '@/api/urls'
import { EditorDialog } from '@/features/editors/EditorDialog'
import { useElementSize } from '@/hooks/useElementSize'
import { formatSize } from '@/lib/format'
import type { Size } from '@/lib/geometry'

import { ASPECTS, buildOps, type Constraints, FREE_ASPECTS } from './crop'
import { CropReadout } from './CropReadout'
import { CropTools } from './CropTools'
import { CORNERS, useCropEdit } from './useCropEdit'

export interface CropEditorProps {
  /** The image in the slot; a derived image reopens on its original with its crop. */
  sha: string
  /** The form's output size. */
  target: Size
  constraints: Constraints
  /**
   * A free crop, for a reference: any shape, kept at its own size, because the model sizes
   * each reference itself (to the output's pixel count).
   */
  free?: boolean
  /** The model scales references down only (FLUX.2 [klein]), so a small crop isn't enlarged. */
  refsKeepSize?: boolean
  /**
   * An image prompt's picture: a free crop that starts square, because the image encoder sees
   * a small square from the middle. Its size doesn't matter, so there's no upscale warning.
   */
  square?: boolean
  /** With the size it's meant for: the resize's, even when the crop was kept at its own. */
  onApply: (image: BlobInfo, out: Size) => void
  onCancel: () => void
}
// Used until the stage has been measured (and in tests, which have no layout).
const FALLBACK_STAGE: Size = { w: 360, h: 480 }

/** Full-screen crop, rotate and flip on the image well (design §8.2, ux.md Phase 4). */
export function CropEditor({
  sha,
  target,
  constraints,
  free: freeCrop = false,
  refsKeepSize = false,
  square = false,
  onApply,
  onCancel,
}: CropEditorProps) {
  const free = freeCrop || square
  const history = useQuery(queries.transform(sha))
  const original = history.data?.original
  const stageRef = useRef<HTMLDivElement>(null)
  const stage = useElementSize(stageRef, FALLBACK_STAGE)
  const editing = useCropEdit({
    stageRef,
    stage,
    saved: history.data,
    target,
    constraints,
    free,
    square,
    refsKeepSize,
  })
  const { natural, edit, img, view, frame, crop, out } = editing

  const apply = useMutation({
    mutationFn: ({ ops }: { ops: Parameters<typeof api.transform>[1]; out: Size }) =>
      api.transform(original ?? sha, ops),
    onSuccess: (image, { out }) => {
      onApply(image, out)
    },
  })

  if (history.error) {
    return (
      <EditorDialog title="Crop" onCancel={onCancel}>
        <p className="editor-loading" role="alert">
          Couldn’t open this image for cropping: {history.error.message}
        </p>
      </EditorDialog>
    )
  }

  const placed: CSSProperties | undefined =
    view && img
      ? { left: view.tx, top: view.ty, width: img.w * view.s, height: img.h * view.s }
      : undefined
  const turned: CSSProperties | undefined =
    natural && edit && view
      ? {
          width: natural.w * view.s,
          height: natural.h * view.s,
          transform: `translate(-50%, -50%) scaleX(${edit.flip ? '-1' : '1'}) rotate(${String(edit.rot)}deg)`,
        }
      : undefined

  return (
    <EditorDialog
      title="Crop"
      onCancel={onCancel}
      action={
        <button
          type="button"
          className="btn small"
          disabled={!edit || !crop || !out || !img || apply.isPending}
          onClick={() => {
            if (edit && crop && out && img) {
              const to = edit.resize ? out : crop
              apply.mutate({ ops: buildOps(edit.rot, edit.flip, crop, img, to), out })
            }
          }}
        >
          {apply.isPending ? 'Applying…' : 'Apply'}
        </button>
      }
    >
      <div
        ref={stageRef}
        className="crop-stage"
        role="application"
        aria-label="Image under the crop frame. Drag to move it; pinch, or press + and −, to zoom."
        tabIndex={0}
        onPointerDown={editing.onPointerDown}
        onPointerMove={editing.onPointerMove}
        onPointerUp={editing.onPointerUp}
        onPointerCancel={editing.onPointerUp}
        onKeyDown={editing.onKeyDown}
        onWheel={(e) => {
          editing.zoom(Math.exp(-e.deltaY / 400), e.nativeEvent.offsetX, e.nativeEvent.offsetY)
        }}
      >
        {original && (
          <div className="crop-image" style={placed}>
            <img
              src={blobUrl(original)}
              alt=""
              draggable={false}
              style={turned}
              onLoad={(e) => {
                const { naturalWidth: w, naturalHeight: h } = e.currentTarget
                if (w && h) editing.setNatural({ w, h })
              }}
            />
          </div>
        )}
        {frame && edit && (
          <div
            className={edit.aspect === 'free' ? 'crop-frame free' : 'crop-frame'}
            style={{ left: frame.x, top: frame.y, width: frame.w, height: frame.h }}
          >
            {CORNERS.map((c) => (
              <span key={c} className={`crop-corner ${c}`} data-corner={c} aria-hidden />
            ))}
          </div>
        )}
        {!edit && <p className="editor-loading">Loading the image…</p>}
      </div>

      <div className="editor-controls">
        <CropTools
          aspects={(free ? FREE_ASPECTS : ASPECTS).map((a) =>
            a.id === 'match' ? { ...a, label: `Match ${formatSize(target.w, target.h)}` } : a,
          )}
          aspect={edit?.aspect}
          disabled={!edit}
          flipped={edit?.flip ?? false}
          resize={free ? undefined : (edit?.resize ?? true)}
          onAspect={editing.setAspect}
          onRotate={editing.rotate}
          onFlip={editing.flip}
          onResize={editing.setResize}
          onReset={editing.reset}
        />
        {out && crop && natural && edit && (
          <CropReadout
            crop={crop}
            out={out}
            natural={natural}
            scale={editing.scale}
            resize={edit.resize}
            free={free}
            square={square}
          />
        )}
        {apply.error && <p role="alert">{apply.error.message}</p>}
      </div>
    </EditorDialog>
  )
}
