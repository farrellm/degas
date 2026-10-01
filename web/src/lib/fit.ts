import type { Fit } from '@/api/types'

export interface FitOption {
  id: Fit
  label: string
}

/** How a picture of another shape is made to fit the output. */
export const FIT_OPTIONS: FitOption[] = [
  { id: 'crop', label: 'Crop to fit' },
  { id: 'pad', label: 'Letterbox' },
  { id: 'stretch', label: 'Stretch' },
]
