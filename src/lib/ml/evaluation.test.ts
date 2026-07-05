import { describe, expect, it } from 'vitest';
import { generateDataset } from './datasets';
import {
	bootstrapInterval,
	confusionMatrix,
	crossValidate,
	f1,
	mae,
	precision,
	r2,
	recall,
	rmse,
	rocAuc,
} from './evaluation';

describe('validation', () => {
	it('scores an easy dataset highly and a label-noise dataset lower', () => {
		expect(crossValidate(generateDataset('moons'), 5, 5)).toBeGreaterThan(0.95);
		expect(crossValidate(generateDataset('outliers'), 5, 5)).toBeLessThan(crossValidate(generateDataset('linear'), 5, 5));
	});
});

describe('classification metrics', () => {
	const m = confusionMatrix([1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 0, 0]);
	it('counts the confusion matrix', () => expect(m).toEqual({ tp: 2, fp: 1, tn: 2, fn: 1 }));
	it('computes precision, recall and F1', () => {
		expect(precision(m)).toBeCloseTo(2 / 3);
		expect(recall(m)).toBeCloseTo(2 / 3);
		expect(f1(m)).toBeCloseTo(2 / 3);
	});
	it('returns zero instead of NaN with no positives predicted', () => {
		expect(f1(confusionMatrix([1, 0], [0, 0]))).toBe(0);
	});
	it('computes AUC, including ties', () => {
		expect(rocAuc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])).toBe(1);
		expect(rocAuc([0, 1], [0.5, 0.5])).toBe(0.5);
	});
});

describe('regression metrics', () => {
	it('computes MAE, RMSE and R2', () => {
		expect(mae([1, 2], [2, 4])).toBe(1.5);
		expect(rmse([0, 0], [3, 4])).toBeCloseTo(Math.sqrt(12.5));
		expect(r2([1, 2, 3], [1, 2, 3])).toBe(1);
		expect(r2([1, 2, 3], [2, 2, 2])).toBe(0);
	});
});

describe('bootstrap', () => {
	it('brackets the mean and narrows with more data', () => {
		const sample = (n: number) => Array.from({ length: n }, (_, i) => (i % 5 === 0 ? 0 : 1));
		const [low, high] = bootstrapInterval(sample(100));
		expect(low).toBeLessThan(0.8);
		expect(high).toBeGreaterThan(0.8);
		const [l2, h2] = bootstrapInterval(sample(1000));
		expect(h2 - l2).toBeLessThan(high - low);
	});
});
