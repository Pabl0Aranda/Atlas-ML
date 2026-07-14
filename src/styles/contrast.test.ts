import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

const css = readFileSync('src/styles/global.css', 'utf8');

const block = (selector: string) => css.slice(css.indexOf(`${selector} {`)).split('}')[0];
const token = (scope: string, name: string) => {
	const match = scope.match(new RegExp(`${name}:\\s*(#[0-9a-f]{6})`));
	if (!match) throw new Error(`Missing six-digit hex token ${name}`);
	return match[1];
};

const luminance = (hex: string) => {
	const [r, g, b] = [1, 3, 5].map((i) => {
		const c = parseInt(hex.slice(i, i + 2), 16) / 255;
		return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
	});
	return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};

const ratio = (a: string, b: string) => {
	const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
	return (hi + 0.05) / (lo + 0.05);
};

const themes = { dark: block(':root'), light: block(":root[data-theme='light']") };
const graphics = ['--viz-series-1', '--viz-series-2', '--viz-series-3', '--viz-series-4', '--viz-muted'];

describe.each(Object.entries(themes))('%s theme', (_, scope) => {
	it.each(graphics)('%s reaches 3:1 on the plot surface', (name) => {
		expect(ratio(token(scope, name), token(scope, '--viz-surface'))).toBeGreaterThanOrEqual(3);
	});

	it('keeps secondary text at 4.5:1 on panels', () => {
		expect(ratio(token(scope, '--viz-muted-ink'), token(scope, '--viz-panel'))).toBeGreaterThanOrEqual(4.5);
	});

	it('keeps text on the accent at 4.5:1', () => {
		expect(ratio(token(css, '--on-accent'), token(css, '--color-accent-600'))).toBeGreaterThanOrEqual(4.5);
	});

	it('keeps the accent visible on the page, for focus rings', () => {
		expect(ratio(token(css, '--color-accent-600'), token(scope, '--viz-surface'))).toBeGreaterThanOrEqual(3);
	});
});
