import katex from 'katex';
import { describe, expect, it } from 'vitest';
import { lines, num, polynomial, powersOf, sum, vector } from './equation';

const renders = (latex: string) => expect(() => katex.renderToString(latex, { displayMode: true, throwOnError: true })).not.toThrow();

describe('num', () => {
	it('rounds to fixed digits and never shows negative zero', () => {
		expect(num(0.7123)).toBe('0.71');
		expect(num(-0.004)).toBe('0.00');
		expect(num(-0)).toBe('0.00');
		expect(num(-1.5, 1)).toBe('-1.5');
	});

	it('names numbers that went wrong instead of printing NaN', () => {
		expect(num(Number.NaN)).toBe('\\text{undefined}');
		expect(num(Infinity)).toBe('\\infty');
		expect(num(-Infinity)).toBe('-\\infty');
	});
});

describe('sum', () => {
	it('pulls signs out of each term', () => {
		expect(sum([0.12, 0.71, -0.3], ['', 'x', 'x^{2}'])).toBe('0.12 + 0.71x - 0.30x^{2}');
		expect(sum([-0.5, -2], ['', 'x'])).toBe('-0.50 - 2.00x');
	});

	it('drops terms that are exactly zero, as lasso makes them', () => {
		expect(sum([0.1, 0, 0.4], ['', 'x', 'x^{2}'])).toBe('0.10 + 0.40x^{2}');
		expect(sum([0, 0.5], ['', 'x'])).toBe('0.50x');
	});

	it('shows 0 when every weight is zero, as at the start of training', () => {
		expect(sum([0, 0, 0], ['', 'x', 'x^{2}'])).toBe('0.00');
	});

	it('keeps a tiny weight that rounds to zero, because it is not exactly zero', () => {
		expect(sum([0.2, 0.001], ['', 'x'])).toBe('0.20 + 0.00x');
	});
});

describe('formulas', () => {
	it('writes polynomials with the right powers', () => {
		expect(powersOf('x', 3)).toEqual(['', 'x', 'x^{2}', 'x^{3}']);
		expect(polynomial([0.1, 0.7])).toBe('\\hat{y} = 0.10 + 0.70x');
	});

	it('produces LaTeX that KaTeX can render', () => {
		renders(polynomial([0.1, -0.7, 0.3, 0, 1.2]));
		renders(lines(`\\nabla L(w) = ${vector(-0.8, 5.4)}`, `w \\leftarrow ${vector(0.1, -0.2)}`));
		renders(polynomial([Number.NaN, Infinity]));
	});
});
