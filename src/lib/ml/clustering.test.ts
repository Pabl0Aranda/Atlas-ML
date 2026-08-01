import { describe, expect, it } from 'vitest';
import { generateDataset } from './datasets';
import { clusterCount, dbscanStep, initDbscan, NOISE, type DbscanParams } from './dbscan';
import type { Dataset } from '../../types/ml';

const at = (x: number, y: number) => ({ x, y, label: 0 });

/** True labels found inside each cluster, ignoring noise. */
const labelsPerCluster = (labels: number[], data: Dataset, count: number) =>
	Array.from({ length: count }, (_, c) => new Set(data.filter((_, i) => labels[i] === c).map((p) => p.label)).size);

describe('DBSCAN', () => {
	const run = (data: Dataset, params: DbscanParams) => {
		let state = initDbscan(data);
		for (let i = 0; i < data.length * 3 && !state.finished; i++) state = dbscanStep(state, data, params);
		return state;
	};

	const chain: Dataset = [at(0, 0), at(0.1, 0), at(0.05, 0.05), at(0.05, -0.05), at(0.19, 0), at(0.9, 0.9)];

	it('leaves isolated points as noise', () => {
		const { labels } = run(chain, { eps: 0.12, minPoints: 3 });
		expect(labels[5]).toBe(NOISE);
	});

	it('claims border points without expanding through them', () => {
		const { labels, core } = run(chain, { eps: 0.12, minPoints: 4 });
		expect(core[4]).toBe(false);
		expect(labels[4]).toBe(labels[0]);
	});

	it('counts a point as its own neighbor', () => {
		const pair = [at(0, 0), at(0.05, 0)];
		expect(clusterCount(run(pair, { eps: 0.1, minPoints: 2 }))).toBe(1);
	});

	it('rescues a point first marked noise when a cluster reaches it', () => {
		const border = [at(0.2, 0), at(0.1, 0), at(0, 0), at(0.05, 0.03), at(0.05, -0.03)];
		const { labels } = run(border, { eps: 0.11, minPoints: 4 });
		expect(labels[0]).toBe(labels[1]);
	});

	it('finds the two moons that k-means cannot separate', () => {
		const moons = generateDataset('moons');
		const state = run(moons, { eps: 0.15, minPoints: 4 });
		expect(clusterCount(state)).toBe(2);
		expect(labelsPerCluster(state.labels, moons, 2)).toEqual([1, 1]);
		expect(state.labels.filter((l) => l === NOISE).length).toBeLessThan(5);
	});

	it('visits every point and then finishes', () => {
		const state = run(chain, { eps: 0.12, minPoints: 3 });
		expect(state.finished).toBe(true);
		expect(state.labels.every((l) => l !== undefined && l !== null)).toBe(true);
	});

	it('marks everything noise when eps is tiny', () => {
		const state = run(chain, { eps: 0.001, minPoints: 2 });
		expect(state.labels.every((l) => l === NOISE)).toBe(true);
		expect(clusterCount(state)).toBe(0);
	});
});
