# Web

- `src/features/<name>/` holds each part of the app (create, editors, results, library,
  session); `src/{api,lib,hooks,components}` are shared and must not import a feature (lint
  enforces it). Import across folders as `@/…`. Server reads go through `queries` in
  `api/queries.ts`, not hand-written query keys.
- Web feature tests (`*.test.tsx`) render the whole `<App />` over a fake `fetch`
  (`src/test/`: `mockApi.ts`, `fixtures.ts`, `render.tsx`; shared setup in `setup.ts`).
  Pure logic has plain unit tests beside its module.
- A custom hook must not return a ref inside its result object: `react-hooks/refs` then
  flags every read of that object during render. Create the ref in the component and pass
  it in.
