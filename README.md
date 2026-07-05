# Atlas ML

Find a machine learning algorithm, watch it run, read the math and inspect the code.
Built with Astro, Starlight, Tailwind CSS and Preact islands.

## Commands

| Command           | Action                                        |
| :---------------- | :-------------------------------------------- |
| `npm install`     | Install dependencies                          |
| `npm run dev`     | Start the dev server at `localhost:4321`      |
| `npm test`        | Run the algorithm unit tests                  |
| `npm run build`   | Build the site and search index into `dist/`  |
| `npm run preview` | Serve the build. Search only works after a build |

## Adding an algorithm

1. Write the pure math in `src/lib/ml/` and test it in `ml.test.ts`.
2. Add a spec in `src/lib/viz/specs/` and register it in `src/lib/viz/registry.ts`. Add its id to `AlgorithmId`.
3. Add a page in `src/content/docs/algorithms/` and link it in the `astro.config.mjs` sidebar.
4. Add `page` to its entry in `src/data/catalog.ts`. Algorithms without a page only need a catalog entry.
