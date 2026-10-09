// Query options for every read of the server, so a key is written once and the cache is
// typed by it: `useQuery(queries.jobs())`, `qc.setQueryData(queries.jobs().queryKey, …)`.

import { infiniteQueryOptions, queryOptions } from '@tanstack/react-query'

import { api } from './client'

/**
 * Every query's key prefix, written once. `queries` builds its keys from these; invalidate a
 * prefix to cover every query of its kind, whatever its arguments.
 */
export const queryKeys = {
  families: ['families'],
  schema: ['schema'],
  assets: ['assets'],
  drive: ['drive'],
  civitaiImport: ['civitai-import'],
  session: ['session'],
  jobs: ['jobs'],
  results: ['results'],
  library: ['library'],
  prompts: ['prompts'],
  transform: ['transform'],
  face: ['face'],
} as const

export const queries = {
  families: () => queryOptions({ queryKey: queryKeys.families, queryFn: api.families }),
  schema: (family: string | undefined, variant: string | undefined, mode: string | undefined) =>
    queryOptions({
      queryKey: [...queryKeys.schema, family, variant, mode],
      queryFn: () => api.schema(family ?? '', variant ?? '', mode ?? ''),
      enabled: !!family && !!variant && !!mode,
      staleTime: Infinity,
    }),
  /** The whole Drive index, fetched once and filtered on the phone. */
  assets: () => queryOptions({ queryKey: queryKeys.assets, queryFn: api.assets }),
  drive: () => queryOptions({ queryKey: queryKeys.drive, queryFn: api.drive }),
  civitaiImport: () =>
    queryOptions({ queryKey: queryKeys.civitaiImport, queryFn: api.civitaiImportState }),
  session: () => queryOptions({ queryKey: queryKeys.session, queryFn: api.session }),
  jobs: () => queryOptions({ queryKey: queryKeys.jobs, queryFn: api.jobs }),
  results: () => queryOptions({ queryKey: queryKeys.results, queryFn: () => api.results() }),
  /** The kept images matching `q`, a page at a time. */
  library: (q: string) =>
    infiniteQueryOptions({
      queryKey: [...queryKeys.library, q],
      queryFn: ({ pageParam }) => api.library(q, pageParam),
      initialPageParam: undefined as string | undefined,
      getNextPageParam: (last) => last.cursor ?? undefined,
    }),
  /** The newest kept images, for the image picker. */
  libraryPicker: () =>
    queryOptions({ queryKey: [...queryKeys.library, 'picker'], queryFn: () => api.library('') }),
  prompts: (q = '') =>
    queryOptions({ queryKey: [...queryKeys.prompts, q], queryFn: () => api.prompts(q) }),
  /** How a derived image was made from its original: never changes for a given blob. */
  transform: (sha: string) =>
    queryOptions({
      queryKey: [...queryKeys.transform, sha],
      queryFn: () => api.getTransform(sha),
      staleTime: Infinity,
    }),
  /** The face a FaceID image prompt reads from a picture; needs the GPU. */
  face: (sha: string, gpuReady: boolean) =>
    queryOptions({
      queryKey: [...queryKeys.face, sha],
      queryFn: () => api.findFace(sha),
      enabled: gpuReady,
      retry: false,
      staleTime: Infinity,
    }),
}
