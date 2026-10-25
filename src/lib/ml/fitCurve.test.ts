import { describe, expect, it } from 'vitest';
import { FIT_NEIGHBORS, fitVerdict, knnFitCurve, sampleSplit } from './fitCurve';

const curve = knnFitCurve('outliers');
const at = (k: number) => curve.find((p) => p.k === k)!;
const bestTest = Math.max(...curve.map((p) => p.test));

describe('sampleSplit', () => {
	it('keeps a quarter of the data for testing', () => {
		const { train, test } = sampleSplit('outliers', 1);
		expect(train).toHaveLength(90);
		expect(test).toHaveLength(30);
	});

	it('gives each seed a different sample', () => {
		expect(sampleSplit('outliers', 1).test).not.toEqual(sampleSplit('outliers', 2).test);
	});
});

describe('knnFitCurve', () => {
	it('has one point per neighbor count', () => {
		expect(curve.map((p) => p.k)).toEqual(FIT_NEIGHBORS);
	});

	it('is perfect on training data at k = 1, because each sample is its own neighbor', () => {
		expect(at(1).train).toBe(1);
	});

	it('matches the figures on the concept page', () => {
		expect(at(1).test).toBeCloseTo(0.819, 2);
		expect(at(25).train).toBeCloseTo(0.903, 2);
		expect(at(25).test).toBeCloseTo(0.882, 2);
		expect(at(85).train).toBeCloseTo(0.714, 2);
		expect(at(85).test).toBeCloseTo(0.686, 2);
	});

	it('peaks in the middle, not at either end', () => {
		const best = curve.findIndex((p) => p.test === bestTest);
		expect(best).toBeGreaterThan(0);
		expect(best).toBeLessThan(curve.length - 1);
	});

	it('has the widest gap at the most flexible end', () => {
		const gaps = curve.map((p) => p.train - p.test);
		expect(Math.max(...gaps)).toBe(gaps[0]);
	});

	it('is repeatable', () => {
		expect(knnFitCurve('outliers')).toEqual(curve);
	});
});

describe('fitVerdict', () => {
	it('names all three regimes on the noisy data', () => {
		expect(fitVerdict(at(1), bestTest)).toBe('overfitting');
		expect(fitVerdict(at(25), bestTest)).toBe('good');
		expect(fitVerdict(at(85), bestTest)).toBe('underfitting');
	});

	it('calls a large gap overfitting even when the test score is the best seen', () => {
		expect(fitVerdict({ k: 1, train: 1, test: 0.8 }, 0.8)).toBe('overfitting');
	});

	it('calls two low scores underfitting', () => {
		expect(fitVerdict({ k: 1, train: 0.6, test: 0.58 }, 0.9)).toBe('underfitting');
	});

	it('does not report overfitting on data where k = 1 generalizes', () => {
		const moons = knnFitCurve('moons');
		const best = Math.max(...moons.map((p) => p.test));
		expect(fitVerdict(moons[0], best)).toBe('good');
	});
});
