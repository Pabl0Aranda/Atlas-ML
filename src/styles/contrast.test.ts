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

	it('keeps text on the accent fill at 4.5:1', () => {
		expect(ratio(token(css, '--on-accent'), token(css, '--accent-fill'))).toBeGreaterThanOrEqual(4.5);
	});

	it('keeps the accent visible on the page, for focus rings and borders', () => {
		expect(ratio(token(scope, '--sl-color-accent'), token(scope, '--sl-color-black'))).toBeGreaterThanOrEqual(3);
	});

	it('keeps body and muted text at 4.5:1 on the page', () => {
		expect(ratio(token(scope, '--sl-color-text'), token(scope, '--sl-color-black'))).toBeGreaterThanOrEqual(4.5);
		expect(ratio(token(scope, '--sl-color-gray-3'), token(scope, '--sl-color-black'))).toBeGreaterThanOrEqual(4.5);
	});

	it.each(['--pipeline-prepare', '--pipeline-train', '--pipeline-validate', '--pipeline-test'])(
		'%s reaches 3:1 on cards',
		(name) => {
			expect(ratio(token(scope, name), token(scope, '--viz-panel'))).toBeGreaterThanOrEqual(3);
		},
	);
});

// Starlight colors links with the high accent in the dark theme and the accent in the light one.
it('keeps links at 4.5:1 on the page in both themes', () => {
	expect(ratio(token(themes.dark, '--sl-color-accent-high'), token(themes.dark, '--sl-color-black'))).toBeGreaterThanOrEqual(4.5);
	expect(ratio(token(themes.light, '--sl-color-accent'), token(themes.light, '--sl-color-black'))).toBeGreaterThanOrEqual(4.5);
});
