import { useQuery } from '@tanstack/react-query'

import { queries } from '@/api/queries'
import { thumbUrl } from '@/api/urls'

/**
 * A FaceID picture: once the GPU has found its face, the aligned crop the model reads, which
 * shows whether it found the right one.
 */
export function FaceTile({ sha, n, gpuReady }: { sha: string; n: string; gpuReady: boolean }) {
  const face = useQuery(queries.face(sha, gpuReady))
  if (face.data) {
    const others = face.data.faces - 1
    return (
      <>
        <img src={thumbUrl(face.data.image.sha256)} alt={`The face in picture ${n}`} />
        {others > 0 && (
          <span className="prompt-picture-note">
            The biggest of {String(face.data.faces)} faces
          </span>
        )}
      </>
    )
  }
  return (
    <>
      <img
        src={thumbUrl(sha)}
        alt={`Picture ${n}`}
        className={face.isFetching ? 'finding' : undefined}
      />
      {face.error && (
        <span className="prompt-picture-note warn" role="alert">
          {face.error.message}
        </span>
      )}
    </>
  )
}
