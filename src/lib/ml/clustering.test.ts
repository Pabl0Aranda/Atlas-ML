import { describe, expect, it } from 'vitest';
import { generateDataset } from './datasets';
import { clusterCount, dbscanStep, initDbscan, NOISE, type DbscanParams } from './dbscan';
import { hierarchicalLabels, hierarchicalStep, initHierarchical, Linkage } from './hierarchical';
import type { Dataset } from '../../types/ml';

const at = (x: number, y: number) => ({ x, y, label: 0 });

/** Share of points grouped the same way as their true label, for two groups. */
function agreement(labels: number[], data: Dataset): number {
	const same = labels.filter((l, i) => (l === labels[0]) === (data[i].label === data[0].label)).length;
	return same / labels.length;
}

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

describe('hierarchical clustering', () => {
	const mergeTo = (data: Dataset, k: number, linkage: Linkage) => {
		let state = initHierarchical(data);
		while (state.clusters.length > k) state = hierarchicalStep(state, data, linkage);
		return state;
	};

	const line = [at(0, 0), at(0.1, 0), at(0.3, 0), at(0.7, 0)];

	it('starts with one cluster per point', () => {
		expect(initHierarchical(line).clusters).toHaveLength(4);
	});

	it('merges the closest pair first', () => {
		const state = hierarchicalStep(initHierarchical(line), line, Linkage.Single);
		expect(state.clusters.find((c) => c.members.length === 2)!.members.sort()).toEqual([0, 1]);
		expect(state.merges[0].height).toBeCloseTo(0.1);
	});

	it('measures clusters by nearest, farthest or average pair', () => {
		const height = (linkage: Linkage) => mergeTo(line, 2, linkage).merges[1].height;
		expect(height(Linkage.Single)).toBeCloseTo(0.2);
		expect(height(Linkage.Complete)).toBeCloseTo(0.3);
		expect(height(Linkage.Average)).toBeCloseTo(0.25);
	});

	it('merges n - 1 times into one cluster, with every point kept once', () => {
		const data = generateDataset('blobs', 30);
		const state = mergeTo(data, 1, Linkage.Average);
		expect(state.merges).toHaveLength(29);
		expect([...state.clusters[0].members].sort((a, b) => a - b)).toEqual(data.map((_, i) => i));
	});

	it('never lowers the merge height with complete linkage', () => {
		const { merges } = mergeTo(generateDataset('blobs', 40), 1, Linkage.Complete);
		for (let i = 1; i < merges.length; i++) expect(merges[i].height).toBeGreaterThanOrEqual(merges[i - 1].height - 1e-12);
	});

	it('follows the moons with single linkage, unlike complete linkage', () => {
		const moons = generateDataset('moons');
		const score = (linkage: Linkage) => {
			const state = mergeTo(moons, 2, linkage);
			return agreement(hierarchicalLabels(state, moons.length), moons);
		};
		expect(score(Linkage.Single)).toBeGreaterThan(score(Linkage.Complete));
	});

	it('labels every point with a cluster index', () => {
		const state = mergeTo(line, 2, Linkage.Single);
		expect(hierarchicalLabels(state, 4)).toEqual([0, 0, 0, 1]);
	});
});
