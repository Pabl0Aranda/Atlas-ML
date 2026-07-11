import { existsSync, readdirSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { CATALOG, FAMILIES, familySidebar, pageSlug, SANDBOX_COUNT } from './catalog';

const docs = 'src/content/docs';

describe('catalog pages', () => {
	it('points every page entry at a file that exists', () => {
		const missing = CATALOG.filter((e) => e.page && !existsSync(`${docs}/${pageSlug(e)}.mdx`)).map((e) => e.name);
		expect(missing).toEqual([]);
	});

	it('lists every algorithm page in a family sidebar', () => {
		const listed = FAMILIES.flatMap((f) => familySidebar(f)).map((i) => i.slug);
		const pages = readdirSync(`${docs}/algorithms`).map((f) => `algorithms/${f.replace('.mdx', '')}`);
		expect(pages.filter((p) => !listed.includes(p))).toEqual([]);
	});

	it('keeps guide pages, which autogenerate, out of family sidebars', () => {
		expect(familySidebar('Testing')).toEqual([]);
	});

	it('counts a sandbox once even when two pages share it', () => {
		const pages = CATALOG.filter((e) => e.page).length;
		expect(SANDBOX_COUNT).toBeLessThan(pages);
		expect(SANDBOX_COUNT).toBe(new Set(CATALOG.flatMap((e) => (e.page ? [e.page.id] : []))).size);
	});
});
