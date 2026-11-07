# AGENTS.md

Guidance for AI agents working on Atlas ML, an interactive machine learning guide. The README introduces the project, and CONTRIBUTING.md covers the structure and workflow for contributors. This file covers how to change the project safely.

## Project Rules

These are the owner's decisions. Don't work around them, and ask before changing one.

- **No backend or database.** The site stays static. Keep state in `localStorage` and share links. Don't propose accounts, server storage or hosted services.
- **The readable engine is the product.** Algorithms run in the browser, in the same TypeScript the pages display. Don't add Python, server-side training or heavy ML libraries.
- **Stable URLs.** Pages are public and share links depend on them, so don't move or rename pages.
- **Honest numbers.** Never write a figure on a page or in a preset note without measuring it first. See [Measuring Claims](#measuring-claims).
- **Commit by feature.** Make one commit per functionality or bug fix, as each one lands. See [Git Workflow](#git-workflow).

## Commands

```bash
npm test          # Unit, sandbox contract, contrast and link tests
npm run check     # Type-check, including .astro and .mdx files
npm run build     # Full static build
npm run dev       # Dev server at localhost:4321
```

Run all three of `npm test`, `npm run check` and `npm run build` before calling work finished. Changes to sandboxes or pages also need checking in a browser, because tests can't see layout.

## Git Workflow

`main` is the only long-lived branch, and it must always pass the tests, type check and build.

1. Branch from `main`, using a short name with a prefix:

   | Prefix | Use it for | Example |
   | --- | --- | --- |
   | `feature/` | New functionality | `feature/learning-paths` |
   | `fix/` | Bug fixes | `fix/compare-phone-nav` |
   | `docs/` | Pages and documentation | `docs/cnn-page` |
   | `chore/` | Tooling and maintenance | `chore/update-astro` |

2. Make one commit per functionality or bug fix, with an imperative subject.
3. Run the checks, then merge into `main` with a fast-forward, rebasing first if `main` has moved.
4. Delete the branch once it's merged.

Don't push, and don't rewrite the history of `main`, unless asked.

## Architecture

```text
src/lib/ml/        Pure algorithms: no DOM, deterministic, tested beside the code
      ↓
src/lib/viz/specs/ An AlgorithmSpec adapts an algorithm to the sandbox
      ↓
src/lib/viz/registry.ts   Loads specs by id, so pages only download what they use
      ↓
src/components/interactive/Sandbox.tsx   One shared UI for every spec
      ↓
src/content/docs/  MDX pages embed <Sandbox client:idle algorithm="id" />
```

Other parts to know:

| Path | Purpose |
| --- | --- |
| `src/data/catalog.ts` | Source of truth for algorithms. Drives the sidebar, catalog, cards and counts |
| `src/lib/pipeline.ts` | Maps each docs page to its workflow stage, for the trail above page titles. New sections need an entry |
| `src/lib/viz/sandboxSetup.ts` | Builds the data for a run, and encodes share links |
| `src/lib/viz/timeline.ts` | Frame history for rewind |
| `src/lib/viz/presets.ts` | Guided sandbox setups and compare pairings |
| `src/components/interactive/controls.tsx`, `PlotCanvas.tsx` | Shared controls and plot. Reuse these in new islands |
| `src/components/docs/AlgorithmSource.astro` | Shows a real file from `src/lib/ml/` on a page |
| `src/styles/global.css` | Design tokens: the slate and indigo palette, `--pipeline-*` stage colors, and `--viz-*` plot colors |

## Rules That Tests Enforce

- **Spec states are immutable.** The timeline keeps earlier frames. A spec that trains in place for speed must provide `snapshot`, as DQN and PPO do. Check this past the first training update, because a short run can miss it.
- **Every spec must finish.** It stays finite and terminates at its default settings.
- **`live` parameters can't affect the run.** Only mark a parameter `live` if `init` and `step` ignore it.
- **`holdout` needs a real classifier.** Only set it when `scene.region` returns class labels.
- **Colors must pass contrast.** Graphics need 3:1 and text needs 4.5:1. White text goes on `--accent-fill`, not `--sl-color-accent`, because indigo-500 is too light for it.
- **Internal links and anchors must exist.**

## Common Tasks

### Add an Algorithm

1. Write the math in `src/lib/ml/`, with a test file beside it. Write the tests first.
2. Add a spec in `src/lib/viz/specs/`, register it, and add its id to `AlgorithmId`.
3. Add a page in `src/content/docs/algorithms/`.
4. Set `page` on its entry in `src/data/catalog.ts`, which also adds it to the sidebar.
5. Add presets if they teach something, after measuring them.

Algorithm pages use these sections, in this order: Intuition & Overview, Mathematical Foundation, Step-by-Step Breakdown, Interactive Sandbox, Code, then Strengths, Limitations & Real-world Trade-offs.

### Add a Page

Add an `.mdx` file to a folder in `src/content/docs/`, and set `sidebar.order` to place it. Link to it from related pages.

### Add an Interactive Island

Put the logic in a pure, tested module, and keep the component thin. Hydrate with `client:visible`, so pages stay light. The exception is `Sandbox`, which uses `client:idle`: it grows when it loads, and doing that early keeps smooth anchor scrolling on target.

Islands are rendered on the server first, so don't touch `window` or `document` while rendering, unless the code only runs after a spec loads, as the sandbox's does.

## Measuring Claims

Pages make specific claims, so they have to be true for this site's data.

- Run the exact setup before quoting it, with a throwaway test or in the browser. Delete any scan files afterward.
- Add a test for figures the page depends on, as `indicators.test.ts` and `fitCurve.test.ts` do.
- Prefer a measured result to a textbook expectation. Data here has surprised us: AdaBoost on XOR beats chance in training, yet stays near 50% on held-out data.
- Figures come from the default seed, so re-measure if a dataset generator changes.
- SVM training accuracy can differ slightly between Node and the browser, from floating-point rounding. Avoid quoting it precisely.
- The feature counts in the README are written by hand, so update them when they change.

## Pitfalls

- **Math in Markdown tables.** A literal `|` splits the cell, so write `\lvert x \rvert`. Long fractions overlap in narrow columns, so use `\dfrac` with short notation.
- **Wide equations.** Split them rather than joining two with `\qquad`.
- **Plot colors.** `hexToRgb` in `canvas.ts` only reads six-digit hex, so keep series colors in that form.
- **Color alone.** Plots pair each class color with a shape. Keep that for accessibility.
- **Input parity.** Anything a mouse can do on a plot must also work by touch and keyboard.
- **Internal links.** The site is served from a subfolder. Plain `/catalog/` links are fine in MDX, because a build step prefixes them, but links in code must use `withBase` from `src/lib/url.ts`.
- **Pages without a sidebar.** They have no menu on phones, so give them their own links, as `compare.astro` does.
- **Metric changes.** The mean only minimizes squared Euclidean distance, so k-means with Manhattan distance must update with medians.
- **Preact state.** Use functional updates in input handlers, because two quick changes can otherwise overwrite each other.
- **Build-time specs.** Thumbnails run `thumbnailSteps` at build time, which ignores `isDone`, so steps must be safe to repeat after a run ends.

## Writing Style

Pages are for beginners, so explain the idea in plain language before the math. Use short sentences, and say honestly when a method's limits or a demo's simplifications matter. All math uses LaTeX.
