import { describe, expect, it } from 'vitest';
import { FEATURED } from '../../data/catalog';
import { sandboxHeading } from './sandboxHeader';

describe('sandboxHeading', () => {
	it('names the algorithm and finds its stage from the page', () => {
		expect(sandboxHeading('knn', '/algorithms/knn/')).toEqual({ name: 'K-Nearest Neighbors', stage: 'train' });
		expect(sandboxHeading('data-cleaning', '/data-preparation/cleaning/')).toEqual({ name: 'Data Cleaning', stage: 'prepare' });
	});

	it('uses the page to tell apart a sandbox shown on two pages', () => {
		expect(sandboxHeading('linear-regression', '/algorithms/linear-regression/')).toEqual({ name: 'Linear Regression', stage: 'train' });
		expect(sandboxHeading('linear-regression', '/testing/regression-metrics/')).toEqual({ name: 'Regression Metrics', stage: 'test' });
	});

	it('works when the site is served from a subfolder', () => {
		expect(sandboxHeading('pca', '/Atlas-ML/algorithms/pca/', '/Atlas-ML/')).toEqual({ name: 'Principal Component Analysis', stage: 'train' });
	});

	it('still names a sandbox embedded on a page outside the catalog', () => {
		const heading = sandboxHeading('polynomial-regression', '/concepts/overfitting-underfitting/');
		expect(heading).toEqual({ name: 'Polynomial Regression', stage: 'validate' });
	});

	it('names a concept sandbox that has no catalog entry', () => {
		expect(sandboxHeading('gradient-descent', '/concepts/gradient-descent/').name).toBe('Gradient Descent');
	});

	it('gives every catalog sandbox a stage on its own page', () => {
		const missing = FEATURED.filter((e) => !sandboxHeading(e.page!.id, `/${e.page!.path ?? `algorithms/${e.page!.id}`}/`).stage);
		expect(missing.map((e) => e.name)).toEqual([]);
	});
});
