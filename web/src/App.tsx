import { useEffect, useState } from 'react'

interface Health {
  status: string
  version: string
}

function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${String(res.status)}`)
        return res.json() as Promise<Health>
      })
      .then(setHealth)
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : String(e))
      })
  }, [])

  return (
    <main>
      <h1>Degas</h1>
      {health && <p>Server {health.version}</p>}
      {error && <p role="alert">Server unreachable: {error}</p>}
    </main>
  )
}

export default App
