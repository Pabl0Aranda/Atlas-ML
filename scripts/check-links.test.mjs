import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';
import { findBrokenLinks } from './check-links.mjs';

let root;
const site = (files) => {
	root = mkdtempSync(join(tmpdir(), 'links-'));
	for (const [path, body] of Object.entries(files)) {
		mkdirSync(dirname(join(root, path)), { recursive: true });
		writeFileSync(join(root, path), body);
	}
	return root;
};
afterEach(() => root && rmSync(root, { recursive: true, force: true }));

describe('findBrokenLinks', () => {
	it('accepts pages, assets, anchors and outside links that exist', () => {
		const dist = site({
			'index.html': '<a href="/Atlas-ML/knn/#sandbox">a</a><link href="/Atlas-ML/_astro/a.css"><a href="https://example.com/x">b</a><a href="#top">c</a><h1 id="top"></h1>',
			'knn/index.html': '<h2 id="sandbox"></h2><a href="../">home</a>',
			'_astro/a.css': '',
		});
		expect(findBrokenLinks(dist, '/Atlas-ML')).toEqual([]);
	});

	it('reports missing pages, files and anchors', () => {
		const dist = site({
			'index.html': '<a href="/Atlas-ML/nope/">a</a><script src="/Atlas-ML/_astro/gone.js"></script><a href="/Atlas-ML/#missing">b</a>',
		});
		expect(findBrokenLinks(dist, '/Atlas-ML').map((b) => b.reason)).toEqual(['missing page or file', 'missing page or file', 'missing anchor']);
	});

	it('reports links that forget the subfolder', () => {
		const dist = site({ 'index.html': '<a href="/catalog/">a</a>', 'catalog/index.html': '' });
		expect(findBrokenLinks(dist, '/Atlas-ML')).toEqual([{ page: '/Atlas-ML/', link: '/catalog/', reason: 'outside /Atlas-ML/' }]);
	});

	it('ignores query strings and decodes encoded paths', () => {
		const dist = site({
			'index.html': '<a href="/compare/?a=svm&amp;b=knn">a</a><a href="/compare/?a=svm&#x26;b=knn">b</a><a href="/catalog/?family=Deep%20Learning">c</a>',
			'compare/index.html': '',
			'catalog/index.html': '',
		});
		expect(findBrokenLinks(dist)).toEqual([]);
	});
});
