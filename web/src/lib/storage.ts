// localStorage, which private browsing and full disks can take away: reads fall back to
// null and writes are dropped, so callers only ever lose what they hoped to remember.

export function readStored(key: string): string | null {
  try {
    return localStorage.getItem(key)
  } catch {
    return null
  }
}

/** Returns whether the value was kept. */
export function writeStored(key: string, value: string): boolean {
  try {
    localStorage.setItem(key, value)
    return true
  } catch {
    return false
  }
}
