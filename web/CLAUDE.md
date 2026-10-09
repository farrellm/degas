# Web

- `src/features/<name>/` holds each part of the app (create, editors, results, library,
  session); `src/{api,lib,hooks,components}` are shared and must not import a feature (lint
  enforces it). Features layer `app → results/library/session → create → editors`: each
  imports only those below it (Create's draft is how the others hand it work), also enforced
  by lint (`.oxlintrc.json`). Import across folders as `@/…`. Server reads go through `queries` in
  `api/queries.ts`, not hand-written query keys.
- Web feature tests (`*.test.tsx`) render the whole `<App />` over a fake `fetch`
  (`src/test/`: `mockApi.ts`, `fixtures.ts`, `render.tsx`; shared setup in `setup.ts`).
  Pure logic has plain unit tests beside its module.
- A custom hook must not return a ref inside its result object: `react-hooks/refs` then
  flags every read of that object during render. Create the ref in the component and pass
  it in.
- A part's stylesheet sits beside it (`features/create/create.css`), but components don't
  import CSS: every sheet is `@import`ed from `styles/index.css`, whose order the cascade
  relies on (sheets win ties with earlier ones). Add a new sheet to that list.
