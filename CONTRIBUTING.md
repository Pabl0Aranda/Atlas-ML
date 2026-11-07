# Contributing to Atlas ML

Thanks for your interest in improving Atlas ML. You can help by [reporting a bug, suggesting a feature or proposing an algorithm](https://github.com/Pabl0Aranda/Atlas-ML/issues/new/choose), or by opening a pull request. Please follow the [Code of Conduct](CODE_OF_CONDUCT.md), and report security problems as described in the [Security Policy](SECURITY.md).

## Project rules

- **No backend or database.** The site stays static, and everything runs in the browser.
- **Readable code is the product.** The algorithms are the same TypeScript the pages display, so keep them clear.
- **Stable URLs.** Pages are public and share links depend on them, so avoid moving or renaming pages.
- **Honest numbers.** Run an example before quoting its figures in a page or preset note.

## Setup

You need Node.js 22.23.4 or later.

```bash
npm install
npm run dev
```

Before opening a pull request, run all three checks:

```bash
npm test
npm run check
npm run build
```

## Branches

`main` is the only long-lived branch and always stays working. Work on a short-lived branch from `main`, named with one of these prefixes:

| Prefix | Use it for | Example |
| :----- | :--------- | :------ |
| `feature/` | New functionality | `feature/learning-paths` |
| `fix/` | Bug fixes | `fix/compare-phone-nav` |
| `docs/` | Pages and documentation | `docs/cnn-page` |
| `chore/` | Tooling and maintenance | `chore/update-astro` |

Make one commit per functionality or bug fix, with a subject in the imperative, such as "Add DBSCAN".

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

## Adding a page

Add an `.mdx` file to a folder in `src/content/docs/`, and the sidebar picks it up. Set `sidebar.order` in the frontmatter to place it.

## Adding an algorithm

1. Write the pure math in `src/lib/ml/`, with a test file beside it.
2. Add a spec in `src/lib/viz/specs/` and register it in `src/lib/viz/registry.ts`. Add its id to `AlgorithmId`.
3. Add a page in `src/content/docs/algorithms/`.
4. Add `page` to its entry in `src/data/catalog.ts`, which also adds it to the sidebar. Algorithms without a page only need a catalog entry.
5. Optionally, add guided presets in `src/lib/viz/presets.ts`. Run each one before writing its note, so the figures are real.

Spec states must be immutable, because the sandbox timeline keeps earlier frames. A spec that trains in place for speed must provide `snapshot`, and the contract tests check this.

## Testing

`npm test` covers more than the algorithms:

- **Contract tests** run every sandbox from start to finish, and check that it stays finite, terminates, and keeps its recorded frames intact.
- **Link tests** fail on any broken internal link or heading anchor.
- **Contrast tests** check the color tokens against accessibility targets.

## Links in code

The site is served from a subfolder. Plain `/catalog/` links are fine in MDX, because a build step prefixes them, but links in code must use `withBase` from `src/lib/url.ts`.

More detail, including common pitfalls, is in [AGENTS.md](AGENTS.md).
