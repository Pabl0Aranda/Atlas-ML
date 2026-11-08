// Builds a Jupyter notebook for every docs page with Python code, into
// public/notebooks/. The notebooks are committed, so Colab can open them from
// GitHub, and a test fails if they fall out of date with the pages.
//
// Usage: node scripts/build-notebooks.mjs

import { mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, sep } from 'node:path';
import { pathToFileURL } from 'node:url';
import { SETUPS } from './notebook-setups.mjs';

export const DOCS = 'src/content/docs';
export const OUT = 'public/notebooks';
const SITE = 'https://pabl0aranda.github.io/Atlas-ML';

/** @param {string} dir @returns {string[]} */
const walk = (dir) =>
	readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]));

/** @param {string} text @param {string} key */
const frontmatter = (text, key) => text.match(new RegExp(`^${key}:\\s*(.+)$`, 'm'))?.[1].trim().replace(/^['"]|['"]$/g, '') ?? '';

/** The Python excerpt on a page, without the fence. Every page has at most one. */
export const pythonIn = (/** @type {string} */ text) => text.replace(/\r\n/g, '\n').match(/^```python[^\n]*\n([\s\S]*?)\n```/m)?.[1];

/** @param {string} source @param {string} id @param {'markdown' | 'code'} type */
function cell(type, source, id) {
	const lines = source.split('\n').map((line, i, all) => (i < all.length - 1 ? `${line}\n` : line));
	return type === 'code'
		? { cell_type: 'code', execution_count: null, id, metadata: {}, outputs: [], source: lines }
		: { cell_type: 'markdown', id, metadata: {}, source: lines };
}

/** @param {string} page The page's content id, such as algorithms/knn. @param {string} text The page's MDX. */
export function notebookFor(page, text) {
	const code = pythonIn(text);
	if (code === undefined) return undefined;
	const setup = SETUPS[page];
	if (!setup) throw new Error(`No notebook setup for ${page}. Add one to scripts/notebook-setups.mjs.`);
	const title = frontmatter(text, 'title');
	const description = frontmatter(text, 'description');
	const cells = [
		cell(
			'markdown',
			`# ${title}\n\n${description}\n\nThis notebook comes from the [${title}](${SITE}/${page}/) page on Atlas ML, an interactive guide to machine learning. Run the cells from top to bottom.`,
			'intro',
		),
	];
	if (setup.install) cells.push(cell('code', setup.install, 'install'));
	if (setup.setup) {
		cells.push(cell('markdown', '## Example data\n\nThe code on the page assumes some data already exists. This cell creates it.', 'setup-intro'));
		cells.push(cell('code', setup.setup, 'setup'));
	}
	cells.push(cell('markdown', '## Code from the page', 'code-intro'));
	cells.push(cell('code', code, 'code'));
	if (setup.demo) {
		cells.push(cell('markdown', '## Try it', 'demo-intro'));
		cells.push(cell('code', setup.demo, 'demo'));
	}
	const notebook = {
		cells,
		metadata: {
			colab: { provenance: [] },
			kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
			language_info: { name: 'python' },
		},
		nbformat: 4,
		nbformat_minor: 5,
	};
	return `${JSON.stringify(notebook, null, 1)}\n`;
}

/** Every notebook the pages call for, as { path under public/notebooks: contents }. */
export function allNotebooks() {
	/** @type {Record<string, string>} */
	const notebooks = {};
	for (const file of walk(DOCS).filter((f) => f.endsWith('.mdx'))) {
		const page = relative(DOCS, file).split(sep).join('/').replace(/\.mdx$/, '');
		const notebook = notebookFor(page, readFileSync(file, 'utf8'));
		if (notebook) notebooks[`${page}.ipynb`] = notebook;
	}
	return notebooks;
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? '').href) {
	rmSync(OUT, { recursive: true, force: true });
	const notebooks = allNotebooks();
	for (const [path, contents] of Object.entries(notebooks)) {
		mkdirSync(dirname(join(OUT, path)), { recursive: true });
		writeFileSync(join(OUT, path), contents);
	}
	console.log(`Wrote ${Object.keys(notebooks).length} notebooks to ${OUT}/.`);
}
