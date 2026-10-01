/**
 * Insert a trigger word into a prompt at `at`, keeping the comma-separated
 * style prompts are usually written in: "a cat" + "filmgrain" → "a cat, filmgrain".
 */
export function insertWord(text: string, at: number, word: string): string {
  const before = text.slice(0, at).trimEnd()
  const after = text.slice(at).trimStart()
  const lead = before === '' ? '' : before.endsWith(',') ? ' ' : ', '
  const tail = after === '' || after.startsWith(',') ? '' : ', '
  return before + lead + word + tail + after
}
