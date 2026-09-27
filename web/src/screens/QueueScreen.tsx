import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, thumbUrl, type Job, type Result } from '../api'
import { SessionPrompt } from '../components/SessionPrompt'

const PHASES: Record<string, string> = {
  copy: 'Copying model',
  load: 'Loading model',
  denoise: 'Denoising',
  decode: 'Decoding',
  encode: 'Encoding',
}

function progressText(job: Job): string {
  const p = job.progress
  if (!p) return job.status === 'running' ? 'Starting…' : ''
  const item =
    job.seeds.length > 1 ? `Image ${String(p.item + 1)}/${String(job.seeds.length)} · ` : ''
  const phase = PHASES[p.phase] ?? p.phase
  if (p.phase === 'copy' && p.steps > 0) {
    return `${item}${phase} ${String(Math.round((100 * p.step) / p.steps))}%`
  }
  return p.steps > 0 ? `${item}${phase} ${String(p.step)}/${String(p.steps)}` : `${item}${phase}`
}

function fraction(job: Job): number {
  const p = job.progress
  if (!p || p.steps === 0) return 0
  const items = job.seeds.length
  const within = p.phase === 'denoise' || p.phase === 'decode' ? p.step / p.steps : 0
  return Math.min(1, (p.item + within) / items)
}

export function QueueScreen() {
  const qc = useQueryClient()
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  const results = useQuery({ queryKey: ['results'], queryFn: () => api.results() })
  const cancel = useMutation({
    mutationFn: api.cancelJob,
    onSettled: () => qc.invalidateQueries({ queryKey: ['jobs'] }),
  })

  if (jobs.isPending) return <p className="muted">Loading…</p>
  if (jobs.error) return <p role="alert">{jobs.error.message}</p>

  const byJob = new Map<string, Result[]>()
  for (const r of results.data?.results ?? []) {
    byJob.set(r.job_id, [...(byJob.get(r.job_id) ?? []), r])
  }

  return (
    <div className="queue">
      <SessionPrompt />
      {jobs.data.length === 0 && <p className="muted">No jobs yet.</p>}
      <ul className="jobs">
        {jobs.data.map((job) => (
          <li key={job.id} className={`job ${job.status}`}>
            <div className="job-head">
              <span className={`badge ${job.status}`}>{job.status}</span>
              <span className="prompt">{String(job.spec.params.prompt ?? '')}</span>
              {(job.status === 'queued' || job.status === 'running') && (
                <button
                  type="button"
                  className="secondary small"
                  onClick={() => {
                    cancel.mutate(job.id)
                  }}
                >
                  Cancel
                </button>
              )}
            </div>
            {job.status === 'running' && (
              <>
                <progress value={fraction(job)} max={1} />
                <div className="muted small">{progressText(job)}</div>
              </>
            )}
            {job.error && <div className="error small">{job.error}</div>}
            {(byJob.get(job.id)?.length ?? 0) > 0 && (
              <div className="thumbs">
                {byJob
                  .get(job.id)
                  ?.toSorted((a, b) => a.item_index - b.item_index)
                  .map((r) => (
                    <img key={r.id} src={thumbUrl(r.blob_sha)} alt={`Seed ${String(r.seed)}`} />
                  ))}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}
