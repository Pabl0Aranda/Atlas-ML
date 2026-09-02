import { describe, expect, it } from 'vitest';
import { confusionMatrix } from './evaluation';
import {
	accuracyOf,
	balancedAccuracy,
	cohenKappa,
	expectedCost,
	fBeta,
	macroF1,
	matrixAccuracy,
	matthews,
	mcnemar,
	multiclassMatrix,
	negativePredictiveValue,
	normalizeMatrix,
	perClass,
	specificity,
	weightedF1,
} from './indicators';

/** The screening example on the Indicators page. */
const screening = { tp: 8, fn: 2, fp: 40, tn: 950 };
/** A model that never predicts the disease, on the same 1,000 people. */
const lazy = { tp: 0, fn: 10, fp: 0, tn: 990 };

describe('binary indicators', () => {
	it('match the worked example on the page', () => {
		expect(accuracyOf(screening)).toBeCloseTo(0.958);
		expect(specificity(screening)).toBeCloseTo(0.9596, 4);
		expect(negativePredictiveValue(screening)).toBeCloseTo(0.9979, 4);
		expect(balancedAccuracy(screening)).toBeCloseTo(0.8798, 4);
		expect(matthews(screening)).toBeCloseTo(0.3536, 3);
		expect(cohenKappa(screening)).toBeCloseTo(0.2637, 3);
	});

	it('give F1 at beta 1, and favor recall as beta grows', () => {
		expect(fBeta(screening, 1)).toBeCloseTo(0.2759, 4);
		expect(fBeta(screening, 2)).toBeCloseTo(0.4545, 4);
	});

	it('expose a model that always predicts the majority class', () => {
		expect(accuracyOf(lazy)).toBeCloseTo(0.99);
		expect(balancedAccuracy(lazy)).toBeCloseTo(0.5);
		expect(matthews(lazy)).toBe(0);
		expect(cohenKappa(lazy)).toBeCloseTo(0);
	});

	it('score a perfect model at 1', () => {
		const perfect = confusionMatrix([1, 1, 0, 0], [1, 1, 0, 0]);
		expect(matthews(perfect)).toBeCloseTo(1);
		expect(cohenKappa(perfect)).toBeCloseTo(1);
	});

	it('score a perfectly wrong model at -1', () => {
		expect(matthews(confusionMatrix([1, 1, 0, 0], [0, 0, 1, 1]))).toBeCloseTo(-1);
	});

	it('handle an empty matrix without dividing by zero', () => {
		const empty = { tp: 0, fn: 0, fp: 0, tn: 0 };
		for (const value of [accuracyOf(empty), balancedAccuracy(empty), matthews(empty), cohenKappa(empty), fBeta(empty, 1)]) {
			expect(Number.isFinite(value)).toBe(true);
		}
	});
});

describe('cost matrix', () => {
	const costs = { fn: 50, fp: 1 };
	const cautious = { tp: 5, fn: 5, fp: 5, tn: 985 };

	it('can reverse the ranking that accuracy gives', () => {
		expect(accuracyOf(cautious)).toBeGreaterThan(accuracyOf(screening));
		expect(expectedCost(screening, costs)).toBe(140);
		expect(expectedCost(cautious, costs)).toBe(255);
	});
});

describe('multi-class matrix', () => {
	const matrix = [[40, 8, 2], [5, 42, 3], [1, 4, 45]];

	it('puts actual classes on rows and predictions on columns', () => {
		expect(multiclassMatrix([0, 0, 1, 2], [0, 1, 1, 0], 3)).toEqual([[1, 1, 0], [0, 1, 0], [1, 0, 0]]);
	});

	it('reads accuracy from the diagonal', () => {
		expect(matrixAccuracy(matrix)).toBeCloseTo(127 / 150);
	});

	it('reads recall along rows and precision down columns', () => {
		const [cat, dog, bird] = perClass(matrix);
		expect(cat.recall).toBeCloseTo(0.8);
		expect(cat.precision).toBeCloseTo(40 / 46);
		expect(dog.precision).toBeCloseTo(42 / 54);
		expect(bird.recall).toBeCloseTo(0.9);
		expect(cat.support).toBe(50);
	});

	it('averages F1 per class', () => {
		expect(macroF1(matrix)).toBeCloseTo(0.847, 3);
	});

	it('lets a large class dominate the weighted average', () => {
		const skewed = [[90, 10], [5, 5]];
		expect(weightedF1(skewed)).toBeGreaterThan(macroF1(skewed));
	});

	it('survives a class with no samples', () => {
		expect(Number.isFinite(macroF1([[1, 0], [0, 0]]))).toBe(true);
	});
});

describe('normalizeMatrix', () => {
	const matrix = [[8, 2], [40, 950]];

	it('by row shows recall on the diagonal', () => {
		expect(normalizeMatrix(matrix, 'row')[0][0]).toBeCloseTo(0.8);
	});

	it('by column shows precision on the diagonal', () => {
		expect(normalizeMatrix(matrix, 'column')[0][0]).toBeCloseTo(8 / 48);
	});

	it('over all cells sums to 1', () => {
		expect(normalizeMatrix(matrix, 'all').flat().reduce((a, b) => a + b, 0)).toBeCloseTo(1);
	});

	it('leaves counts alone, and empty rows at zero', () => {
		expect(normalizeMatrix(matrix, 'none')).toEqual(matrix);
		expect(normalizeMatrix([[0, 0], [1, 1]], 'row')[0]).toEqual([0, 0]);
	});
});

describe('McNemar test', () => {
	it('uses only the samples where the models disagree', () => {
		expect(mcnemar(12, 4)).toBeCloseTo(3.0625);
	});

	it('is zero when nothing separates the models', () => {
		expect(mcnemar(0, 0)).toBe(0);
		expect(mcnemar(5, 5)).toBe(0);
	});
});
