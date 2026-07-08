# AGENTS.md

## Project Context

This project is an **Astro-based web application** designed to provide interactive visualization, documentation, and conceptual breakdowns of core Machine Learning (ML) algorithms. The goal is to build a performant, content-focused platform where users can learn about ML models and see them in action visually.

## Tech Stack & Architecture

* **Framework:** Astro (SSG/SSR, static-first with minimal client-side JS burden)
* **Language:** TypeScript
* **Styling:** Tailwind CSS
* **Interactive Components:** Native web components or lightweight client-side frameworks (e.g., Preact/React) hydrated only on interactive visualization pages via Astro directives (`client:visible` or `client:idle`).

---

## Developer Collaboration Guidelines (for AI Agents)

### 1. Architectural Rules & Code Conventions

* **Astro Components (`.astro`):** Use standard Astro components for layout, docs formatting, static visual cards, and routing. Keep client-side scripts inside `.astro` files to a minimum.
* **Client Hydration:** Hydrate interactive components (canvas-based visualizers, parameter sliders, matrix inspectors) strictly using Astro `client:*` directives. Do not send client JS for static documentation pages.
* **TypeScript & Typing:** Maintain strict type definitions for algorithm datasets, matrix inputs, model parameters, and step-by-step execution states. Place global domain types under `src/types/`.
* **Modular Algorithm Engine:** Keep ML mathematical calculations, data generation, and model execution logic separated from UI components. Put core algorithm logic in pure TS files under `src/lib/ml/` so they can be unit-tested without DOM dependencies.

---

## Repository Structure & Core Modules

```text
├── src/
│   ├── components/
│   │   ├── docs/          # Markdown/MDX components, mathematical notation, callouts
│   │   ├── interactive/   # Visualizers (canvas, SVG controls, data point charts)
│   │   └── ui/            # Common UI elements (buttons, sidebars, sliders)
│   ├── content/
│   │   └── docs/          # MDX documentation entries for algorithms
│   ├── layouts/           # Main site, docs, and full-screen visualization layouts
│   ├── lib/
│   │   └── ml/            # Pure TypeScript implementations of ML algorithms & linear algebra helpers
│   ├── pages/             # File-based routes for pages and interactive demos
│   └── types/             # Shared TypeScript interfaces (DataPoint, ModelMetrics, Hyperparameters)
└── public/                # Static assets and pre-computed datasets

```

---

## Functional Scope & Key Modules

### 1. Interactive Visualizers (`src/components/interactive/`)

Interactive modules must render algorithms step-by-step or in real time based on user parameter adjustments:

* **Supervised Learning:**
* *Linear/Polynomial Regression:* Dynamic gradient descent step visualization, loss curve fitting, residual error plots.
* *Decision Trees:* Interactive tree depth builder, feature split decision boundary visualization.
* *KNN (k-Nearest Neighbors):* 2D spatial Voronoi diagram boundary shifts as $k$ scales.
* *Support Vector Machines (SVM):* Margin maximization, support vector highlights, dual problem hyperplanes.


* **Unsupervised Learning:**
* *K-Means Clustering:* Centroid movement per iteration, Voronoi partition updating.
* *PCA (Principal Component Analysis):* Variance projection onto vector axes in 2D/3D.


* **Neural Networks:** Simple perceptron decision boundaries and multi-layer feedforward activation flows.

### 2. Interactive Controls Standard

Every visualization interface must support standard state controls:

* Dataset selectors (Linear, Moons, Circles, Noise/Outliers generation).
* Hyperparameter controls (Learning rate $\alpha$, Epochs/Iterations, $k$ value, regularizers).
* Execution triggers (Step Forward, Play/Pause, Reset).

### 3. Documentation System (`src/content/docs/`)

* MDX format for clear separation of prose, mathematical formulations (LaTeX/KaTeX), and embedded interactive components.
* Standard structure per algorithm document:

1. **Intuition & Overview**
2. **Mathematical Foundation** (Objective functions, gradients, cost formulations)
3. **Step-by-Step Breakdown**
4. **Interactive Sandbox** (Embedded visualizer)
5. **Strengths, Limitations & Real-world Trade-offs**



---

## Agent Instructions for Code Generation & Modifications

* When implementing or modifying an ML algorithm, write the core math as pure, deterministic TypeScript functions inside `src/lib/ml/` first.
* Keep visualization rendering performance high; prefer `<canvas>` or optimized SVG rendering for datasets exceeding 500 points.
* Ensure all mathematical notation in documentation uses standard LaTeX formatting for rendering compatibility.
