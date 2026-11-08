import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { describe, expect, it } from 'vitest';
import { allNotebooks, DOCS, notebookFor, OUT, pythonIn } from './build-notebooks.mjs';
import { SETUPS } from './notebook-setups.mjs';

const walk = (dir) =>
	readdirSync(dir, { withFileTypes: true }).flatMap((e) => (e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]));
const pages = walk(DOCS)
	.filter((f) => f.endsWith('.mdx'))
	.map((f) => ({ id: relative(DOCS, f).split(sep).join('/').replace(/\.mdx$/, ''), text: readFileSync(f, 'utf8') }));
const withPython = pages.filter((p) => pythonIn(p.text) !== undefined);
const notebooks = allNotebooks();

describe('notebooks', () => {
	it('match the pages, so run `node scripts/build-notebooks.mjs` after editing Python code', () => {
		const stale = Object.entries(notebooks)
			.filter(([path, contents]) => !existsSync(join(OUT, path)) || readFileSync(join(OUT, path), 'utf8').replace(/\r\n/g, '\n') !== contents)
			.map(([path]) => path);
		expect(stale).toEqual([]);
	});

	it('leave no notebook behind for a page that lost its code', () => {
		const committed = existsSync(OUT) ? walk(OUT).map((f) => relative(OUT, f).split(sep).join('/')) : [];
		expect(committed.filter((path) => !(path in notebooks))).toEqual([]);
	});

	it('exist for every page with Python code, each with buttons that point at it', () => {
		expect(withPython.length).toBeGreaterThan(40);
		for (const page of withPython) {
			expect(notebooks[`${page.id}.ipynb`], page.id).toBeDefined();
			expect(page.text, page.id).toContain(`<NotebookLinks page="${page.id}" />`);
		}
	});

	it('keep a setup only for pages that have code', () => {
		const ids = new Set(withPython.map((p) => p.id));
		expect(Object.keys(SETUPS).filter((id) => !ids.has(id))).toEqual([]);
	});

	it('are valid version 4 notebooks with the page code in a code cell', () => {
		const page = withPython.find((p) => p.id === 'algorithms/knn');
		const notebook = JSON.parse(notebookFor(page.id, page.text));
		expect(notebook.nbformat).toBe(4);
		const code = notebook.cells.find((c) => c.id === 'code');
		expect(code.cell_type).toBe('code');
		expect(code.source.join('')).toBe(pythonIn(page.text));
		expect(new Set(notebook.cells.map((c) => c.id)).size).toBe(notebook.cells.length);
	});
});
