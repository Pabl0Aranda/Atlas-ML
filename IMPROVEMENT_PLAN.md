# Atlas ML improvement plan

Audited 2026-11-06 from the source code, the existing `dist/` build, `npm test` and `astro check`.
The site was not opened in a browser for this audit. Lighthouse scores, real mobile behaviour and
runtime performance are therefore still unmeasured. Effort figures are rough estimates for one developer.

## 1. Direction

Atlas ML already has a strong core: a sandbox that runs the same code the page shows you. That is
the thing to build on. The site currently reads as a well-made Starlight docs site, and it can
become a place people return to in order to learn.

Three goals, in priority order:

1. **Make the sandbox work for everyone.** It should work on touch and keyboard, and runs should be shareable.
2. **Turn a reference into a learning product.** Add guided paths, comparison, and self-checks.
3. **Give it an identity.** The name "Atlas" suggests a map, and the design can follow that idea.

## 2. Where the project stands

| Area | Current state |
| --- | --- |
| Stack | Astro 7, Starlight, Tailwind 4, Preact islands, KaTeX, Pagefind |
| Catalog | 64 entries, of which 28 have full pages and 36 link out to Wikipedia |
| Sandboxes | 27 registered specs behind one shared `Sandbox` component |
| Tests | 67 unit tests pass across 7 files. `src/lib/viz` has no tests |
| Type check | `astro check` reports 0 errors across 63 files |
| Client JS | Small and split per algorithm. The largest app chunk is `Sandbox` at about 18 KB |
| Delivery | No CI, no `site` URL, no deployment configuration |

Coverage is uneven, and this is the biggest content gap:

| Family | Entries | With a page |
| --- | --- | --- |
| Deep Learning | 15 | 2 |
| Unsupervised Learning | 13 | 2 |
| Supervised Learning | 12 | 5 |
| Statistical Learning | 10 | 8 |
| Reinforcement Learning | 7 | 4 |

The home page presents deep learning as a full category, but 13 of its 15 entries leave the site.

## 3. Fix first

These are defects or risks confirmed in the code.

### Functional

- **Class B cannot be added on touch or keyboard.** `Sandbox.tsx` only reads `event.shiftKey`, so phone
  users can add class A samples only. Datasets with more than two classes cannot receive the others at all.
  Replace this with a visible class picker.
- **The plot has no keyboard path.** The canvas is `role="img"` with a click handler and no focus.
  Add a focusable plot with arrow-key cursor movement and Enter to place a sample.
- **Hover readouts have no touch equivalent.** The probe updates on pointer move and hides on leave.
  Make a tap pin the probe, and set `touch-action` deliberately so plots do not fight page scrolling.
- **Every parameter change restarts the run.** Sliders fire on input, so dragging resets progress repeatedly.
  Parameters that do not invalidate the model, such as display options, should apply live.
- **The live caption may be noisy.** The caption is `aria-live="polite"` and changes on every tick while playing.
  Announce only on pause, on Step and on completion.

### Accessibility of colour

Measured contrast ratios against `--viz-surface`. Graphics need 3:1 and small text needs 4.5:1.

| Token | Theme | Ratio | Status |
| --- | --- | --- | --- |
| `--viz-series-4` | Light | 2.11 | Fails |
| `--viz-muted` (wall tiles, neutral nodes) | Light | 2.50 | Fails |
| `--viz-series-3` | Light | 2.74 | Fails |
| `--viz-series-2` | Light | 3.12 | Passes narrowly |
| White on accent `#2a78d6` (buttons) | Both | 4.42 | Fails for small text |

Class shapes already back up colour on the plot, which is a good decision to keep.

### Project hygiene

- **Commit the current work.** Tracked files are modified, and new pages such as Elastic Net are untracked.
- **Resolve `AGENTS.md`.** It is listed in `.gitignore` yet tracked by git, so pick one.
- **Set `site` in `astro.config.mjs`.** Without it there is no sitemap and no canonical URL.
- **Add a social image.** Pages declare `summary_large_image` but no `og:image` exists.
- **Clear the deprecation warning.** `markdown.remarkPlugins` and `rehypePlugins` are deprecated in this Astro version.
- **Remove hard-coded counts.** `index.astro` excludes "Regression Metrics" by name to compute the sandbox count.
  Derive the count from the data instead.
- **Reduce the steps for adding an algorithm.** Pages are wired by hand in the type, registry, sidebar and catalog.
  Generate the sidebar from `catalog.ts`.

## 4. Visual design

### Concept: a field atlas

Treat machine learning as a territory. Families are regions, algorithms are locations, and the workflow
is the route. This gives the site a recognisable look without decoration for its own sake:

- A faint contour-line texture in the hero, generated as SVG at build time.
- Coordinates and scale-bar details on plots, using the existing mono font.
- A colour per family, used the way a map legend uses colour.

### Proposed palette

The current look is neutral grey plus default blue. The proposal moves to blue-tinted ink, warm paper
and a teal accent. All ratios below were computed for this plan.

| Role | Dark | Light | Ratio dark | Ratio light |
| --- | --- | --- | --- | --- |
| Surface | `#10161f` | `#fbfaf7` | n/a | n/a |
| Panel | `#171f2b` | `#f2f0ea` | n/a | n/a |
| Ink | `#e8ecf2` | `#141a23` | 15.31 | 16.74 |
| Secondary text, on panel | `#a7b1c0` | `#4d5664` | 7.65 | 6.51 |
| Muted graphics, on surface | `#7d8899` | `#7a8291` | 5.06 | 3.71 |
| Accent, on surface | `#2dd4bf` | `#0f766e` | 9.75 | 5.24 |
| Text on accent | `#10161f` | `#ffffff` | 9.75 | 5.47 |

Data and family colours share one set, so the site needs only five hues.

| Hue | Dark | Light | Lowest ratio dark | Lowest ratio light | Plot series | Family |
| --- | --- | --- | --- | --- | --- | --- |
| Blue | `#5b9df0` | `#1f6fd0` | 5.94 | 4.34 | 1 | Supervised |
| Orange | `#f0834f` | `#c9501c` | 6.36 | 3.96 | 2 | Reinforcement |
| Violet | `#a78bfa` | `#7048d6` | 6.09 | 5.14 | 3 | Unsupervised |
| Amber | `#e0a526` | `#a06a00` | 7.56 | 4.04 | 4 | Statistical |
| Rose | `#f0749a` | `#c42d62` | 6.05 | 4.73 | Not used | Deep Learning |

"Lowest ratio" is the worse of surface and panel. Green is left out on purpose, because it would sit
too close to the teal accent.

Implementation notes:

- **Add an `--on-accent` token.** Text on accent is hard-coded white in `Sandbox.tsx`, `index.astro`
  and `.page-jump-primary`. White fails on the bright dark-mode teal.
- **Keep series colours as six-digit hex.** `hexToRgb` in `canvas.ts` parses that format only.
- **Set the accent per theme.** The two themes need different teals, so define the accent in a theme override.
- **Test colour-blind separation before adopting.** Contrast was measured, but colour-vision simulation
  was not. Blue and violet are the pair most likely to merge.

If you prefer the current calmer look, keep the neutrals and take only the corrected series and `--on-accent` token.

### Typography and layout

- Keep Inter, Space Grotesk and JetBrains Mono. The set is good, and a fourth font is not needed.
- Use larger hero type with a tighter measure, since the hero has the most room for personality.
- Add a family-coloured edge and small icon to cards, the page eyebrow and the catalog chips.
- Give the sandbox more width. `--sl-content-width` is 50rem, which squeezes the plot beside its 14rem controls.
  Let sandboxes break out of the column on wide screens, and add a full-screen toggle.

### Home page

- **Replace the static hero with a live one.** Run a small looping demo, with the SVG as the no-JS and reduced-motion fallback.
- **Add "The map of ML".** An interactive diagram of families and their algorithms, with entries that have pages highlighted.
- **Reword the stats.** "Algorithms indexed" overstates depth while so many entries are links. Show pages and sandboxes first.

## 5. Sandbox upgrades

Ordered by value for effort. Most fit the existing `AlgorithmSpec` contract.

1. **Shareable state.** Encode algorithm, dataset, seed and parameters in the URL. Datasets are already
   generated from a seed, so this is cheap and enables links, embeds and lesson presets.
2. **Step back and a timeline scrubber.** Keep a bounded history of states. Being able to rewind is what makes step-by-step learning work.
3. **Guided presets.** Add buttons such as "Learning rate too high" or "Overfit with k = 1", each with a short explanation.
4. **Drawing tools.** Add a class picker, drag to paint samples, drag to move, erase and clear.
5. **Train and test view.** Show held-out points and both scores, linking the sandbox to the site's own validation lessons.
6. **Compare mode.** Show two sandboxes side by side on the same data, such as Ridge against Lasso, or KNN against SVM.
7. **Decision-confidence shading.** Fade region colour by certainty for models that output probabilities.
8. **Loss landscape view.** Show gradient descent moving over a contour plot. The Gradient Descent page has no sandbox today.
9. **Export.** Offer PNG download, dataset CSV, and a generated scikit-learn snippet that matches the current parameters.
10. **Bring your own data.** Accept a two-feature CSV, parsed locally in the browser.

The six guide and concept pages without a sandbox are good candidates too. Bias-variance and
gradient descent are the two most valuable.

## 6. New site features

- **Learning paths.** Offer ordered routes such as "ML in a weekend", "From regression to neural networks" and "Reinforcement learning".
  Add next and previous steps and an estimated reading time.
- **Progress tracking.** Store visited pages and completed checks in `localStorage`. No accounts or backend are needed.
- **Knowledge checks.** Add two to four questions per page, including prediction prompts such as "What happens if k rises?"
- **"Which algorithm should I use?"** An interactive decision guide based on task, data size and need for interpretability.
- **Comparison table.** Filter algorithms by task, training cost, scalability and interpretability. This needs new fields in `catalog.ts`.
- **Glossary with hover definitions.** Link terms such as "regularization" across pages, and include them in search.
- **Prerequisites and related links.** Add both to each page, and show catalog difficulty levels.
- **Math notation toggle.** Offer plain-language explanations beside key equations for beginners.
- **Spanish translation.** Starlight has built-in i18n. Start with the interface and the most-visited pages.
- **Embeddable sandboxes.** A minimal route for teachers and bloggers, which depends on shareable state.
- **Offline support.** Add a small service worker later, since the site is static and suits offline study.

## 7. Content roadmap

Choose pages by how clearly they animate and how often people look for them.

| Wave | Pages | Why |
| --- | --- | --- |
| 1 | Random Forest, Gradient Boosting, AdaBoost, DBSCAN, Hierarchical Clustering | Well-known methods that reuse the tree and 2D plot code |
| 2 | Feedforward Networks (merge with MLP), CNN, RNN, LSTM, Autoencoders | Fills the weakest family. Needs new visual types |
| 3 | Transformers, t-SNE, UMAP, Isolation Forest, Hidden Markov Models | Popular, but harder to show honestly |
| 4 | VAE, GAN, Diffusion Models, Actor-Critic methods | Toy versions only, and labelled as such |

Short comparison pages may serve XGBoost, LightGBM and CatBoost better than full sandboxes, since
their differences are mostly engineering rather than visible behaviour.

Non-2D models need new scene types. Plan a matrix or heatmap for attention and convolution, an image
grid, and a sequence strip. Build those once in `src/types/viz.ts` before starting wave 2.

Concept pages are also thin at about 30 lines each. Add overfitting, regularization, loss functions,
the curse of dimensionality, and data leakage.

## 8. Engineering

- **CI on every push.** Run tests, `astro check` and the build in GitHub Actions.
- **Deployment.** Use Cloudflare Pages, Netlify or GitHub Pages, with preview builds per branch.
- **Spec contract tests.** Loop over all registry entries and assert that `init`, `step`, `scene` and `metrics`
  stay finite and that each run terminates.
- **End-to-end tests.** Use Playwright for sandbox controls, theme switching and catalog filters, with axe for accessibility.
- **Link checking.** Validate internal links and catalog `page` references at build time.
- **Performance measurement.** Set budgets in Lighthouse CI. Sandboxes run on the main thread, and regions cost
  72 × 72 classifier calls per redraw. Measure the slower models, and move them to a Web Worker only if it is justified.
- **Reduce `any`.** Card props are cast with `as any` in the pages, and `AlgorithmSpec` defaults to `any` state.
- **Privacy-friendly analytics.** Plausible or similar would show which pages to expand, without a cookie banner.

## 9. Phases

| Phase | Outcome | Main work | Effort |
| --- | --- | --- | --- |
| 0. Foundation | Safe to change and publish | Commit work, CI, deploy, `site`, social image, deprecation fix | 2 to 3 days |
| 1. Inclusive sandbox | Works on phone and keyboard | Class picker, keyboard plot, tap probe, contrast fixes, tests | 1 week |
| 2. Identity | Recognisable look | Palette, family colours, cards, hero, wider sandbox | 1 to 2 weeks |
| 3. Better learning tools | Sandbox teaches actively | URL state, rewind, presets, train and test view, compare mode | 2 to 3 weeks |
| 4. Learning product | Reasons to return | Paths, progress, checks, glossary, decision guide | 3 to 4 weeks |
| 5. Content depth | Fewer Wikipedia exits | Waves 1 and 2, then new scene types | 4 to 6 weeks |
| 6. Reach | Wider audience | Spanish, embeds, CSV upload, offline support, waves 3 and 4 | Ongoing |

Start with phases 0 and 1. Phase 3's shareable state should land before phase 4, because paths and
presets depend on it.

## 10. How to measure success

- All sandboxes pass a manual phone and keyboard check, and axe reports no serious issues.
- Lighthouse reaches 90 or more in every category on the home page and a sandbox page.
- At least half of the catalog has full pages after phase 5.
- Analytics show sandbox interaction rate, path completion and returning visitors.

## 11. Decisions for the owner

1. **Audience.** This plan assumes students and self-taught beginners. A practitioner audience would
   put the comparison table and decision guide ahead of paths and quizzes.
2. **Palette.** Choose the field atlas palette, or keep the neutrals with corrected contrast.
3. **Language.** Decide whether Spanish is a priority, because translation raises the cost of every later page.
4. **Hosting and domain.** Both are needed before the `site` value can be set.

## 12. Not recommended for now

- **User accounts and a backend.** `localStorage` covers progress, and the static build is a real advantage.
- **Server-side training or Python in the browser.** The readable TypeScript engine is the site's distinguishing feature.
- **An AI tutor.** It adds cost and accuracy risk before the core learning loop exists.
- **A 3D rendering library.** Add one only if PCA in 3D proves its value, since it would outweigh the current JS budget.
