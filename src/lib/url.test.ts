import { describe, expect, it } from 'vitest';
import { rehypeBaseLinks, withBase } from './url';

describe('withBase', () => {
	it('leaves paths alone at the root of a domain', () => {
		expect(withBase('/catalog/', '/')).toBe('/catalog/');
		expect(withBase('/', '/')).toBe('/');
	});

	it('adds a subfolder, with or without a trailing slash', () => {
		expect(withBase('/catalog/', '/Atlas-ML')).toBe('/Atlas-ML/catalog/');
		expect(withBase('/catalog/', '/Atlas-ML/')).toBe('/Atlas-ML/catalog/');
	});

	it('keeps the home link inside the subfolder', () => {
		expect(withBase('/', '/Atlas-ML')).toBe('/Atlas-ML/');
	});

	it('keeps queries and anchors', () => {
		expect(withBase('/compare/?a=knn', '/Atlas-ML')).toBe('/Atlas-ML/compare/?a=knn');
		expect(withBase('/validation/overview/#good-practices', '/Atlas-ML')).toBe('/Atlas-ML/validation/overview/#good-practices');
	});
});

describe('rehypeBaseLinks', () => {
	const link = (href: string) => ({ type: 'element', tagName: 'a', properties: { href }, children: [] });
	const hrefAfter = (href: string, base = '/Atlas-ML') => {
		const node = link(href);
		rehypeBaseLinks({ base })({ type: 'root', children: [{ type: 'element', tagName: 'p', properties: {}, children: [node] }] });
		return node.properties.href;
	};

	it('prefixes internal links, including nested ones', () => {
		expect(hrefAfter('/catalog/')).toBe('/Atlas-ML/catalog/');
	});

	it('leaves external links, anchors and relative links alone', () => {
		expect(hrefAfter('https://example.com/a')).toBe('https://example.com/a');
		expect(hrefAfter('#code')).toBe('#code');
		expect(hrefAfter('../other/')).toBe('../other/');
	});

	it('leaves protocol-relative links alone', () => {
		expect(hrefAfter('//example.com/a')).toBe('//example.com/a');
	});

	it('does nothing at the root of a domain', () => {
		expect(hrefAfter('/catalog/', '/')).toBe('/catalog/');
	});
});
