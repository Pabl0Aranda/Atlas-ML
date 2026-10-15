# Atlas ML

Find a machine learning algorithm, watch it run, read the math and inspect the code.
Built with Astro, Starlight, Tailwind CSS and Preact islands.

## Commands

| Command           | Action                                        |
| :---------------- | :-------------------------------------------- |
| `npm install`     | Install dependencies                          |
| `npm run dev`     | Start the dev server at `localhost:4321`      |
| `npm test`        | Run the unit and sandbox contract tests       |
| `npm run check`   | Type-check the project                        |
| `npm run build`   | Build the site and search index into `dist/`  |
| `npm run preview` | Serve the build. Search only works after a build |

## Project structure

| Folder | What lives there |
| :----- | :--------------- |
| `src/content/docs/` | The pages, in MDX. Folder names become URLs |
| `src/pages/` | Pages built in code: home, catalog and compare |
| `src/data/catalog.ts` | Every algorithm, which also drives the sidebar and counts |
| `src/lib/ml/` | Pure algorithm and math code with its tests, and the source shown on pages |
| `src/lib/viz/` | Sandbox logic: specs, presets, links, timeline and drawing |
| `src/components/interactive/` | Client-side islands, such as the sandbox and demos |
| `src/components/ui/`, `docs/`, `overrides/` | Static cards, doc helpers and Starlight overrides |
| `src/styles/global.css` | Design tokens and shared styles |
| `src/types/` | Shared TypeScript types |

Page URLs are public, and share links depend on them, so avoid moving pages.

## Adding a page

Add an `.mdx` file to a folder in `src/content/docs/`, and the sidebar picks it up. Set `sidebar.order` in the frontmatter to place it. `npm test` checks every internal link and anchor.

## Adding an algorithm

1. Write the pure math in `src/lib/ml/`, with a test file beside it.
2. Add a spec in `src/lib/viz/specs/` and register it in `src/lib/viz/registry.ts`. Add its id to `AlgorithmId`.
3. Add a page in `src/content/docs/algorithms/`.
4. Add `page` to its entry in `src/data/catalog.ts`, which also adds it to the sidebar. Algorithms without a page only need a catalog entry.
5. Optionally, add guided presets in `src/lib/viz/presets.ts`. Run each one before writing its note, so the figures are real.

Spec states must be immutable, because the sandbox timeline keeps earlier frames. A spec that trains in place for speed must provide `snapshot`, and the contract tests check this.

## Deployment

Set `SITE_URL` to the public origin when building, for example `SITE_URL=https://example.com npm run build`. It turns on the sitemap, canonical URLs and social image tags.
