import { describe, expect, it } from 'vitest';
import { generateDataset } from './datasets';
import { mean } from './linalg';
import {
	dropDuplicates,
	fitMinMax,
	fitStandard,
	flagOutliers,
	imputeMedian,
	iqrBounds,
	kFold,
	oneHot,
	quantile,
	standardDeviation,
	trainTestSplit,
} from './preprocessing';

describe('cleaning', () => {
	it('computes quantiles and Tukey fences', () => {
		expect(quantile([1, 2, 3, 4], 0.5)).toBe(2.5);
		expect(iqrBounds([1, 2, 3, 4, 5], 1.5)).toEqual([-1, 7]);
	});
	it('flags the injected outliers and few others', () => {
		const data = generateDataset('dirty');
		const flagged = flagOutliers(data).filter(Boolean).length;
		expect(flagged).toBeGreaterThanOrEqual(6);
		expect(flagged).toBeLessThan(16);
	});
	it('imputes medians and drops duplicates', () => {
		expect(imputeMedian([1, null, 3, 100])).toEqual([1, 3, 3, 100]);
		const p = { x: 1, y: 2, label: 0 };
		expect(dropDuplicates([p, { ...p }, { ...p, label: 1 }])).toHaveLength(2);
	});
});

describe('processing', () => {
	it('scales to the expected ranges', () => {
		const values = generateDataset('wide').map((p) => p.x);
		const min = values.map(fitMinMax(values));
		expect(Math.min(...min)).toBe(0);
		expect(Math.max(...min)).toBe(1);
		const z = values.map(fitStandard(values));
		expect(mean(z)).toBeCloseTo(0);
		expect(standardDeviation(z)).toBeCloseTo(1);
	});
	it('survives constant columns', () => {
		expect(fitStandard([2, 2])(2)).toBe(0);
		expect(fitMinMax([2, 2])(2)).toBe(0);
	});
	it('one-hot encodes', () => {
		expect(oneHot(['b', 'a', 'b'])).toEqual({ categories: ['a', 'b'], rows: [[0, 1], [1, 0], [0, 1]] });
	});
});

describe('separating', () => {
	const data = generateDataset('imbalanced');
	const positives = (flags: boolean[]) => data.filter((p, i) => flags[i] && p.label === 1).length;
	it('hits the ratio and is deterministic', () => {
		const flags = trainTestSplit(data, 0.25, false);
		expect(flags.filter(Boolean)).toHaveLength(30);
		expect(flags).toEqual(trainTestSplit(data, 0.25, false));
	});
	it('stratification preserves the class share', () => {
		const total = data.filter((p) => p.label === 1).length;
		expect(positives(trainTestSplit(data, 0.25, true))).toBe(Math.round(total * 0.25));
	});
	it('k-fold covers every sample exactly once', () => {
		const folds = kFold(103, 5);
		expect(folds.flat().sort((a, b) => a - b)).toEqual(Array.from({ length: 103 }, (_, i) => i));
		expect(Math.max(...folds.map((f) => f.length)) - Math.min(...folds.map((f) => f.length))).toBeLessThanOrEqual(1);
	});
});
