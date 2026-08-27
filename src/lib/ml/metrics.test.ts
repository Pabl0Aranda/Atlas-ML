import { describe, expect, it } from 'vitest';
import { generateDataset } from './datasets';
import { dbscanStep, initDbscan } from './dbscan';
import { PlaneMetric } from './distance';
import { hierarchicalStep, initHierarchical, Linkage } from './hierarchical';
import { initKMeans, kMeansStep, nearestCentroid } from './kmeans';
import { knnClassify, knnLeaveOneOut } from './knn';
import type { Dataset } from '../../types/ml';

const at = (x: number, y: number, label = 0) => ({ x, y, label });

describe('KNN with a distance metric', () => {
	// (0.6, 0.6) is nearer in a straight line, but (1, 0) is nearer along a grid.
	const data: Dataset = [at(0.6, 0.6, 0), at(1, 0, 1)];

	it('defaults to Euclidean', () => {
		expect(knnClassify(data, 1, 0, 0)).toBe(0);
	});

	it('can change the prediction', () => {
		expect(knnClassify(data, 1, 0, 0, PlaneMetric.Manhattan)).toBe(1);
	});

	it('scores leave-one-out with the same metric', () => {
		const moons = generateDataset('moons');
		const score = knnLeaveOneOut(moons, 1, PlaneMetric.Chebyshev);
		expect(score).toBeGreaterThan(0.9);
		expect(score).toBeLessThanOrEqual(1);
	});
});

describe('DBSCAN with a distance metric', () => {
	const run = (metric?: PlaneMetric) => {
		const data: Dataset = [at(0, 0), at(0.1, 0.1)];
		let state = initDbscan(data);
		while (!state.finished) state = dbscanStep(state, data, { eps: 0.12, minPoints: 2, metric });
		return state.clusters;
	};

	it('measures the neighborhood with that metric', () => {
		expect(run()).toBe(0);
		expect(run(PlaneMetric.Chebyshev)).toBe(1);
	});
});

describe('hierarchical clustering with a distance metric', () => {
	// The diagonal pair is closest in a straight line, and the horizontal pair along a grid.
	const data: Dataset = [at(0, 0), at(0.3, 0.3), at(-0.45, 0)];

	const firstHeight = (metric?: PlaneMetric) => hierarchicalStep(initHierarchical(data), data, Linkage.Single, metric).merges[0];

	it('merges by that metric', () => {
		expect(firstHeight().height).toBeCloseTo(Math.hypot(0.3, 0.3));
		expect(firstHeight(PlaneMetric.Manhattan).height).toBeCloseTo(0.45);
	});
});

describe('k-medians', () => {
	const data: Dataset = [at(0, 0), at(0.1, 0), at(0.2, 0), at(0.9, 0)];
	const start = { ...initKMeans(data, 1), centroids: [{ x: 0.5, y: 0 }], assignments: [0, 0, 0, 0] };

	it('moves centroids to the mean by default', () => {
		expect(kMeansStep(start, data).centroids[0].x).toBeCloseTo(0.3);
	});

	it('moves centroids to the median with Manhattan distance, which resists outliers', () => {
		expect(kMeansStep(start, data, PlaneMetric.Manhattan).centroids[0].x).toBeCloseTo(0.15);
	});

	it('assigns samples by that metric', () => {
		const centroids = [{ x: 0.6, y: 0.6 }, { x: 1, y: 0 }];
		expect(nearestCentroid(centroids, 0, 0)).toBe(0);
		expect(nearestCentroid(centroids, 0, 0, PlaneMetric.Manhattan)).toBe(1);
	});

	it('never raises its cost, and converges', () => {
		const blobs = generateDataset('blobs');
		let state = initKMeans(blobs, 4, 11, PlaneMetric.Manhattan);
		for (let i = 0; i < 50 && !state.converged; i++) state = kMeansStep(state, blobs, PlaneMetric.Manhattan);
		expect(state.converged).toBe(true);
		const costs = state.inertiaHistory;
		for (let i = 1; i < costs.length; i++) expect(costs[i]).toBeLessThanOrEqual(costs[i - 1] + 1e-9);
	});
});
