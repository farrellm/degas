/** Join class names, leaving out the ones whose condition didn't hold. */
export function cx(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(' ')
}
