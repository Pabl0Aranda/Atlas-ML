# Atlas ML

An interactive guide to machine learning. Find an algorithm, watch it run step by step, read the math, and inspect the exact code behind every animation.

Everything runs in the browser, on the same TypeScript the pages display, so there is no backend or database.

## Features

- **32 interactive sandboxes.** Play, step and rewind a run, change settings, add your own samples, and share the exact setup as a link.
- **26 algorithms explained,** across supervised, unsupervised, statistical, deep and reinforcement learning, in a catalog of 64 entries.
- **Compare mode.** Run two algorithms side by side on identical data.
- **Train and test view.** Hold out data on classifiers to see overfitting directly.
- **Guided presets.** One-click setups that make a single idea visible, with notes based on real runs.
- **The full workflow.** Guides to data preparation, validation and testing, including indicators and validation matrices.
- **Foundations.** Overfitting and underfitting, bias and variance, gradient descent, and a section on distance metrics.
- **Works for everyone.** Sandboxes support touch and keyboard, and plots back up color with shape.

New visitors should begin at the Start Here page, which also explains every sandbox control.

## Tech stack

| Area | Tools |
| :--- | :---- |
| Site | Astro, Starlight |
| Interactive parts | Preact islands, canvas and SVG |
| Styling | Tailwind CSS |
| Math | KaTeX |
| Search | Pagefind |
| Tests | Vitest, `astro check` |

## Getting started

You need Node.js 22.23 or later.

```bash
npm install
npm run dev
```

| Command | Action |
| :------ | :----- |
| `npm run dev` | Start the dev server at `localhost:4321` |
| `npm test` | Run the unit, sandbox contract and link tests |
| `npm run check` | Type-check the project |
| `npm run build` | Build the site and search index into `dist/` |
| `npm run preview` | Serve the build. Search only works after a build |

CI runs the tests, type check and build on every push.

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

## Testing

`npm test` covers more than the algorithms:

- **Contract tests** run every sandbox from start to finish, and check that it stays finite, terminates, and keeps its recorded frames intact.
- **Link tests** fail on any broken internal link or heading anchor.
- **Contrast tests** check the color tokens against accessibility targets.

## Adding a page

Add an `.mdx` file to a folder in `src/content/docs/`, and the sidebar picks it up. Set `sidebar.order` in the frontmatter to place it.

## Adding an algorithm

1. Write the pure math in `src/lib/ml/`, with a test file beside it.
2. Add a spec in `src/lib/viz/specs/` and register it in `src/lib/viz/registry.ts`. Add its id to `AlgorithmId`.
3. Add a page in `src/content/docs/algorithms/`.
4. Add `page` to its entry in `src/data/catalog.ts`, which also adds it to the sidebar. Algorithms without a page only need a catalog entry.
5. Optionally, add guided presets in `src/lib/viz/presets.ts`. Run each one before writing its note, so the figures are real.

Spec states must be immutable, because the sandbox timeline keeps earlier frames. A spec that trains in place for speed must provide `snapshot`, and the contract tests check this.

## Deployment

The site builds to static files, so any static host works. Set `SITE_URL` to the public origin when building, which turns on the sitemap, canonical URLs and social image tags:

```bash
SITE_URL=https://example.com npm run build
```
