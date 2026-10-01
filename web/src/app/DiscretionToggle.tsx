import { setDiscretion, useDiscretion } from '../discretion'

/** The header switch for discretion mode: covers images and prompts until tapped. */
export function DiscretionToggle() {
  const on = useDiscretion()
  return (
    <button
      type="button"
      className="discretion-toggle"
      aria-label="Cover images"
      aria-pressed={on}
      onClick={() => {
        setDiscretion(!on)
      }}
    >
      <svg viewBox="0 0 20 20" aria-hidden>
        {on ? (
          <path d="M2.5 8.5c3.5 4.5 11.5 4.5 15 0M5.5 11.5 4 13.5M10 12.8v2.5M14.5 11.5l1.5 2" />
        ) : (
          <>
            <path d="M2 10s3-5.5 8-5.5 8 5.5 8 5.5-3 5.5-8 5.5S2 10 2 10z" />
            <circle cx="10" cy="10" r="2.5" />
          </>
        )}
      </svg>
    </button>
  )
}
