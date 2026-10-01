// What a job starts from. The ids are the server's; the labels are Create's mode chips.

export const MODE_LABELS: Record<string, string> = {
  t2i: 'From text',
  t2v: 'From text',
  i2i: 'From image',
  i2v: 'From image',
  edit: 'Edit',
  inpaint: 'Inpaint',
  outpaint: 'Outpaint',
}

/** The order the mode chips come in; any mode not listed goes after these. */
export const MODE_ORDER = Object.keys(MODE_LABELS)

/** The modes that start from an image. */
export const SOURCE_MODES = new Set(['i2i', 'i2v', 'edit', 'inpaint', 'outpaint'])

export const MEDIA_LABELS = { image: 'Image', video: 'Video' } as const
