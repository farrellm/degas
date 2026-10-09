import { useMutation } from '@tanstack/react-query'
import { useEffect, useEffectEvent, useState } from 'react'

import { api } from '@/api/client'
import type { Asset, Params, TraceId } from '@/api/types'
import { type Source, toSource } from '@/lib/image'

import {
  type ControlUnit,
  DEFAULT_EDGES,
  edgeDetail,
  edgeParams,
  matchingModel,
  photoOf,
  sameSize,
  underlay,
} from './control'

// How long the Detail slider rests before the edges are traced again.
const RETRACE_MS = 400

/**
 * A ControlNet unit's picture and the control image traced from it on the GPU: tracing it,
 * replacing the picture (which takes the area along where it can), and retracing edges as
 * the Detail slider moves.
 */
export function useControlTrace({
  unit,
  controlnets,
  onChange,
}: {
  unit: ControlUnit
  /** The family's ControlNets, to choose one that reads a new trace. */
  controlnets: Asset[]
  onChange: (update: (unit: ControlUnit) => ControlUnit) => void
}) {
  // What became of the area when the picture or its trace last changed.
  const [note, setNote] = useState<string | null>(null)
  const [detail, setDetail] = useState(() =>
    unit.trace?.id === 'canny' ? edgeDetail(unit.trace.params) : edgeDetail(DEFAULT_EDGES),
  )

  // Trace the unit's picture. The area stays when the trace is the size it was painted at.
  const trace = useMutation({
    mutationFn: async (v: { id: TraceId; params: Params; from: Source }) => {
      const out = await api.trace(v.id, v.from.sha, v.params)
      return { ...v, image: toSource(out.image) }
    },
    onMutate: () => setNote(null),
    onSuccess: (r) => {
      if (unit.area && !sameSize(unit.image, r.image))
        setNote('The new trace is another size, so the area was cleared.')
      onChange((u) => ({
        ...u,
        image: r.image,
        trace: { id: r.id, params: r.params, from: r.from },
        area: sameSize(u.image, r.image) ? u.area : null,
        model: u.model || (matchingModel(controlnets, r.id)?.path ?? ''),
      }))
    },
  })

  // A new picture (picked, cropped, or Create's source): the area follows a crop of the
  // same picture, and an existing trace is made again from the new picture.
  const repicture = useMutation({
    mutationFn: async (v: { photo: Source; before: ControlUnit; cropped: boolean }) => {
      const { photo, before, cropped } = v
      let area: string | null = null
      let message: string | null = null
      const old = photoOf(before)
      const areaOnPhoto = !!old && underlay(before) === old.sha
      if (before.area && cropped && old && areaOnPhoto) {
        try {
          const moved = await api.remapMask(before.area, old.sha, photo.sha)
          if (moved.empty) message = 'The crop left out the whole area.'
          else {
            area = moved.sha256
            message = 'The area moved with the crop.'
          }
        } catch {
          message = 'The area couldn’t follow the crop. Paint it again.'
        }
      } else if (before.area) {
        message = 'A new picture clears the area.'
      }
      if (!before.trace) return { image: photo, trace: null, area, message }
      try {
        const out = await api.trace(before.trace.id, photo.sha, before.trace.params)
        return {
          image: toSource(out.image),
          trace: { ...before.trace, from: photo },
          area,
          message,
        }
      } catch (e) {
        const why = e instanceof Error ? e.message : String(e)
        return { image: photo, trace: null, area, message: `Using the picture as it is. ${why}` }
      }
    },
    onMutate: () => setNote(null),
    onSuccess: (r) => {
      onChange((u) => ({ ...u, image: r.image, trace: r.trace, area: r.area }))
      setNote(r.message)
    },
  })

  const busy = trace.isPending || repicture.isPending
  const error = trace.error ?? repicture.error

  // Edges: trace again once the Detail slider rests.
  const cannyTrace = unit.trace?.id === 'canny' ? unit.trace : null
  const retrace = useEffectEvent((from: Source) =>
    trace.mutate({ id: 'canny', params: edgeParams(detail), from }),
  )
  useEffect(() => {
    if (!cannyTrace || edgeDetail(cannyTrace.params) === detail) return
    const id = setTimeout(() => retrace(cannyTrace.from), RETRACE_MS)
    return () => clearTimeout(id)
  }, [detail, cannyTrace])

  const takePicture = (photo: Source, cropped: boolean) =>
    repicture.mutate({ photo, before: unit, cropped })

  return {
    busy,
    error,
    note,
    clearNote: () => setNote(null),
    detail,
    setDetail,
    trace: trace.mutate,
    takePicture,
  }
}
