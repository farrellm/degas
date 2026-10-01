// Shapes shared by the editors and the forms that open them.

export interface Size {
  w: number
  h: number
}

export interface Rect {
  x: number
  y: number
  w: number
  h: number
}

/** How the (rotated) image sits on the stage: `s` stage px per image px, top-left at `tx, ty`. */
export interface View {
  s: number
  tx: number
  ty: number
}

/** Aspect ratios closer than this are the same shape: sizes are rounded to whole pixels. */
const RATIO_TOLERANCE = 0.01

/** Whether ratio `a` is off ratio `b` by more than rounding: the picture will need fitting. */
export const ratioDiffers = (a: number, b: number) => Math.abs(a / b - 1) > RATIO_TOLERANCE

/** Whether ratio `a` is, to within rounding, ratio `b`. */
export const ratioMatches = (a: number, b: number) => Math.abs(a / b - 1) < RATIO_TOLERANCE
