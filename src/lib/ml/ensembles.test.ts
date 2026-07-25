import { describe, expect, it } from 'vitest';
import { generateDataset } from './datasets';
import { boostingClassify, boostingProbability, boostingStep, initBoosting } from './gradientBoosting';
import { accuracy } from './knn';
import { forestClassify, forestStep, initForest } from './randomForest';
import type { Dataset } from '../../types/ml';

const repeat = <S>(n: number, s: S, f: (s: S) => S) => {
	for (let i = 0; i < n; i++) s = f(s);
	return s;
};

const moons = generateDataset('moons');

describe('random forest', () => {
	const grow = (trees: number, maxDepth = 5) => repeat(trees, initForest(moons), (s) => forestStep(s, moons, maxDepth));

	it('is deterministic', () => {
		expect(JSON.stringify(grow(5))).toBe(JSON.stringify(grow(5)));
	});

	it('gives each tree a different bootstrap sample', () => {
		const { trees } = grow(2);
		expect(JSON.stringify(trees[0])).not.toBe(JSON.stringify(trees[1]));
	});

	it('samples features at each split, so stumps use both axes', () => {
		const features = new Set(grow(20, 1).trees.flatMap((t) => (t.kind === 'split' ? [t.feature] : [])));
		expect(features).toEqual(new Set(['x', 'y']));
	});

	it('scores out-of-bag samples, which is harder than the training set', () => {
		const forest = grow(25);
		const training = accuracy(moons, (x, y) => forestClassify(forest, x, y));
		expect(forest.oobAccuracy).toBeGreaterThan(0.9);
		expect(forest.oobAccuracy).toBeLessThanOrEqual(training);
		expect(forest.oobHistory).toHaveLength(25);
	});

	it('leaves about a third of the samples out of each tree', () => {
		const votes = grow(1).oobVotes.filter((v) => v.some((c) => c > 0)).length;
		expect(votes / moons.length).toBeGreaterThan(0.25);
		expect(votes / moons.length).toBeLessThan(0.5);
	});

	it('predicts a class before any tree exists', () => {
		expect(forestClassify(initForest(moons), 0, 0)).toBe(0);
	});
});

describe('gradient boosting', () => {
	const params = { learningRate: 0.3, maxDepth: 2 };
	const fit = (rounds: number, p = params) => repeat(rounds, initBoosting(moons), (s) => boostingStep(s, moons, p));

	it('starts from the log-odds of the positive class', () => {
		const skewed: Dataset = [{ x: 0, y: 0, label: 1 }, { x: 1, y: 0, label: 0 }, { x: 0, y: 1, label: 0 }, { x: 1, y: 1, label: 0 }];
		const start = initBoosting(skewed);
		expect(start.base).toBeCloseTo(Math.log(1 / 3));
		expect(boostingProbability(start, 0, 0)).toBeCloseTo(0.25);
	});

	it('lowers the log loss as trees are added', () => {
		const { lossHistory } = fit(30);
		expect(lossHistory).toHaveLength(31);
		for (let i = 1; i < lossHistory.length; i++) expect(lossHistory[i]).toBeLessThanOrEqual(lossHistory[i - 1] + 1e-9);
	});

	it('fits moons', () => {
		const model = fit(40);
		expect(accuracy(moons, (x, y) => boostingClassify(model, x, y))).toBeGreaterThan(0.97);
	});

	it('learns more slowly with a smaller learning rate', () => {
		expect(fit(10, { ...params, learningRate: 0.05 }).loss).toBeGreaterThan(fit(10).loss);
	});

	it('stays finite on a class-pure dataset', () => {
		const pure: Dataset = [{ x: 0, y: 0, label: 1 }, { x: 1, y: 1, label: 1 }];
		const model = repeat(5, initBoosting(pure), (s) => boostingStep(s, pure, params));
		expect(Number.isFinite(model.loss)).toBe(true);
		expect(boostingClassify(model, 0, 0)).toBe(1);
	});
});
