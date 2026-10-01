import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';

const DOCS = 'src/content/docs';
const PAGES = 'src/pages';

const walk = (dir: string): string[] =>
	readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
		const path = join(dir, entry.name);
		return entry.isDirectory() ? walk(path) : [path];
	});

const sources = [...walk(DOCS), ...walk(PAGES), ...walk('src/components')].filter((f) => /\.(mdx|astro)$/.test(f));

/** Every internal link in a file, such as /validation/overview/#good-practices. */
const linksIn = (text: string) =>
	[...text.matchAll(/(?:\]\(|href="|link: ')(\/[^)"'\s?]*)/g)].map((m) => m[1]).filter((link) => link !== '/');

const docFile = (path: string) => join(DOCS, `${path.replace(/^\/|\/$/g, '')}.mdx`);
const pageFile = (path: string) => join(PAGES, `${path.replace(/^\/|\/$/g, '')}.astro`);

/** Matches how Starlight turns a heading into an anchor. */
const slug = (heading: string) =>
	heading
		.toLowerCase()
		.replace(/[$`*]/g, '')
		.replace(/[^\p{L}\p{N}\s-]/gu, '')
		.trim()
		.replace(/\s+/g, '-');

const anchorsIn = (text: string) => new Set([...text.matchAll(/^#{2,6}\s+(.+)$/gm)].map((m) => slug(m[1])));

function brokenLinks(text: string): string[] {
	return linksIn(text).filter((link) => {
		const [path, anchor] = link.split('#');
		if (!path || path === '/') return false;
		const target = [docFile(path), pageFile(path)].find(existsSync);
		if (!target) return true;
		return Boolean(anchor) && target.endsWith('.mdx') && !anchorsIn(readFileSync(target, 'utf8')).has(anchor);
	});
}

describe('the link checker', () => {
	it('accepts real pages and anchors', () => {
		expect(brokenLinks('[a](/catalog/) [b](/validation/overview/#good-practices)')).toEqual([]);
	});

	it('catches a missing page, in Markdown and in HTML', () => {
		expect(brokenLinks('[a](/nope/) <a href="/also-nope/">b</a>')).toEqual(['/nope/', '/also-nope/']);
	});

	it('catches a missing anchor', () => {
		expect(brokenLinks('[a](/validation/overview/#nope)')).toEqual(['/validation/overview/#nope']);
	});

	it('finds links across the site', () => {
		expect(sources.flatMap((f) => linksIn(readFileSync(f, 'utf8'))).length).toBeGreaterThan(50);
	});
});

describe('internal links', () => {
	it.each(sources.map((f) => relative('.', f).replaceAll('\\', '/')))('%s has no broken links', (file) => {
		expect(brokenLinks(readFileSync(file, 'utf8'))).toEqual([]);
	});
});
