import { readdirSync } from 'node:fs';
import { join, relative } from 'node:path';
import { describe, expect, it } from 'vitest';
import { STAGES, stageFor } from './pipeline';

const DOCS = 'src/content/docs';
const pages = (dir: string): string[] =>
	readdirSync(dir, { withFileTypes: true }).flatMap((entry) =>
		entry.isDirectory() ? pages(join(dir, entry.name)) : [relative(DOCS, join(dir, entry.name)).replace(/\\/g, '/').replace(/\.mdx?$/, '')],
	);

describe('stageFor', () => {
	it('places every docs page except Start Here in a stage', () => {
		const unplaced = pages(DOCS).filter((id) => id !== 'start-here' && !stageFor(id));
		expect(unplaced).toEqual([]);
	});

	it('keeps Start Here outside the workflow', () => {
		expect(stageFor('start-here')).toBeUndefined();
	});

	it('maps sections and concepts to the right stage', () => {
		expect(stageFor('data-preparation/cleaning')).toBe('prepare');
		expect(stageFor('algorithms/knn')).toBe('train');
		expect(stageFor('distance-metrics/geometric/')).toBe('train');
		expect(stageFor('validation/cross-validation')).toBe('validate');
		expect(stageFor('testing/regression-metrics')).toBe('test');
		expect(stageFor('concepts/gradient-descent')).toBe('train');
		expect(stageFor('concepts/overfitting-underfitting')).toBe('validate');
	});

	it('lists the four stages in workflow order', () => {
		expect(STAGES.map((s) => s.label)).toEqual(['Prepare', 'Train', 'Validate', 'Test']);
	});
});
