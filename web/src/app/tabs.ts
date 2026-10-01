export const TABS = ['Create', 'Results', 'Library'] as const
export type Tab = (typeof TABS)[number]

/** Where a link into the app points: `/?tab=results`, `/?sheet=session` (notifications). */
export function linkTarget(url: string): { tab?: Tab; session: boolean } {
  const params = new URL(url, location.origin).searchParams
  const tab = TABS.find((t) => t.toLowerCase() === params.get('tab'))
  return { tab, session: params.get('sheet') === 'session' }
}
