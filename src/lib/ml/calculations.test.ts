import { describe, expect, it } from 'vitest';
import { generateDataset } from './datasets';
import { bestSplit, buildTree, gini, splitsAtLevel, splitStats } from './decisionTree';
import { knnClassify, nearestNeighbors } from './knn';
import { classifyNaiveBayes, explainNaiveBayes, fitNaiveBayes } from './naiveBayes';

const xor = generateDataset('xor');
const moons = generateDataset('moons');

describe('splitStats', () => {
	it('weighs each side by its share of the samples, and gain is the drop from the parent', () => {
		const data = [
			{ x: -1, y: 0, label: 0 },
			{ x: -0.5, y: 0, label: 0 },
			{ x: 0.5, y: 0, label: 1 },
			{ x: 1, y: 0, label: 0 },
		];
		const stats = splitStats(data, 'x', 0);
		expect(stats.left).toEqual({ samples: 2, gini: 0 });
		expect(stats.right).toEqual({ samples: 2, gini: 0.5 });
		expect(stats.weighted).toBeCloseTo(0.25);
		expect(stats.gain).toBeCloseTo(gini(data) - 0.25);
	});

	it('agrees with the score bestSplit uses to choose', () => {
		const best = bestSplit(moons)!;
		expect(splitStats(moons, best.feature, best.threshold).weighted).toBeCloseTo(best.score, 10);
	});
});

describe('splitsAtLevel', () => {
	it('lists the root split at level 1, with every training sample', () => {
		const tree = buildTree(moons, 1);
		const [root] = splitsAtLevel(tree, moons, 1);
		expect(splitsAtLevel(tree, moons, 1)).toHaveLength(1);
		expect(root.path).toEqual([]);
		expect(root.samples).toBe(moons.length);
		expect(root.left.samples + root.right.samples).toBe(moons.length);
	});

	it('lists one row per node split at the newest level, with the rules that lead to it', () => {
		const tree = buildTree(xor, 2);
		const rows = splitsAtLevel(tree, xor, 2);
		expect(rows.length).toBeGreaterThan(0);
		expect(rows.length).toBeLessThanOrEqual(2);
		for (const row of rows) {
			expect(row.path).toHaveLength(1);
			expect(row.left.samples + row.right.samples).toBe(row.samples);
		}
		expect(rows.reduce((s, r) => s + r.samples, 0)).toBeLessThanOrEqual(xor.length);
	});

	it('is empty before the first split', () => {
		expect(splitsAtLevel(buildTree(xor, 0), xor, 0)).toEqual([]);
	});
});

describe('nearestNeighbors', () => {
	it('returns the k closest samples in order, and agrees with the KNN vote', () => {
		const neighbors = nearestNeighbors(moons, 7, 0.1, 0.2);
		expect(neighbors).toHaveLength(7);
		for (let i = 1; i < neighbors.length; i++) expect(neighbors[i].distance).toBeGreaterThanOrEqual(neighbors[i - 1].distance);
		const votes = [0, 1].map((c) => neighbors.filter((n) => n.label === c).length);
		const winner = votes[0] === votes[1] ? neighbors[0].label : votes.indexOf(Math.max(...votes));
		expect(knnClassify(moons, 7, 0.1, 0.2)).toBe(winner);
	});
});

describe('explainNaiveBayes', () => {
	const model = fitNaiveBayes(moons);

	it('gives posteriors that sum to 1 and pick the same class as the classifier', () => {
		for (const [x, y] of [[0.1, 0.2], [-0.6, 0.4], [0.7, -0.5]]) {
			const rows = explainNaiveBayes(model, x, y, ['x', 'y']);
			expect(rows.reduce((s, r) => s + r.posterior, 0)).toBeCloseTo(1, 10);
			const best = rows.reduce((a, b) => (b.posterior > a.posterior ? b : a));
			expect(best.label).toBe(classifyNaiveBayes(model, x, y, ['x', 'y']));
		}
	});

	it('uses only the prior when no features are in play', () => {
		const rows = explainNaiveBayes(model, 0.1, 0.2, []);
		for (const row of rows) {
			expect(row.likelihood).toEqual({});
			expect(row.posterior).toBeCloseTo(model.find((c) => c.label === row.label)!.prior, 10);
		}
	});
});
