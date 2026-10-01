// Query options for every read of the server, so a key is written once and the cache is
// typed by it: `useQuery(queries.jobs())`, `qc.setQueryData(queries.jobs().queryKey, …)`.

import { infiniteQueryOptions, queryOptions } from '@tanstack/react-query'

import { api } from './client'

export const queries = {
  families: () => queryOptions({ queryKey: ['families'], queryFn: api.families }),
  schema: (family: string | undefined, variant: string | undefined, mode: string | undefined) =>
    queryOptions({
      queryKey: ['schema', family, variant, mode],
      queryFn: () => api.schema(family ?? '', variant ?? '', mode ?? ''),
      enabled: !!family && !!variant && !!mode,
      staleTime: Infinity,
    }),
  /** The whole Drive index, fetched once and filtered on the phone. */
  assets: () => queryOptions({ queryKey: ['assets'], queryFn: api.assets }),
  drive: () => queryOptions({ queryKey: ['drive'], queryFn: api.drive }),
  civitaiImport: () =>
    queryOptions({ queryKey: ['civitai-import'], queryFn: api.civitaiImportState }),
  session: () => queryOptions({ queryKey: ['session'], queryFn: api.session }),
  jobs: () => queryOptions({ queryKey: ['jobs'], queryFn: api.jobs }),
  results: () => queryOptions({ queryKey: ['results'], queryFn: () => api.results() }),
  /** The kept images matching `q`, a page at a time. */
  library: (q: string) =>
    infiniteQueryOptions({
      queryKey: ['library', q],
      queryFn: ({ pageParam }) => api.library(q, pageParam),
      initialPageParam: undefined as string | undefined,
      getNextPageParam: (last) => last.cursor ?? undefined,
    }),
  /** The newest kept images, for the image picker. */
  libraryPicker: () =>
    queryOptions({ queryKey: ['library', 'picker'], queryFn: () => api.library('') }),
  prompts: (q = '') => queryOptions({ queryKey: ['prompts', q], queryFn: () => api.prompts(q) }),
  /** How a derived image was made from its original: never changes for a given blob. */
  transform: (sha: string) =>
    queryOptions({
      queryKey: ['transform', sha],
      queryFn: () => api.getTransform(sha),
      staleTime: Infinity,
    }),
  /** The face a FaceID image prompt reads from a picture; needs the GPU. */
  face: (sha: string, gpuReady: boolean) =>
    queryOptions({
      queryKey: ['face', sha],
      queryFn: () => api.findFace(sha),
      enabled: gpuReady,
      retry: false,
      staleTime: Infinity,
    }),
}

/** Key prefixes to invalidate: each covers every query of its kind, whatever its arguments. */
export const queryKeys = {
  assets: ['assets'],
  drive: ['drive'],
  session: ['session'],
  jobs: ['jobs'],
  results: ['results'],
  library: ['library'],
  prompts: ['prompts'],
} as const
