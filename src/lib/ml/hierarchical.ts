import type { Dataset } from '../../types/ml';

export const Linkage = { Single: 0, Complete: 1, Average: 2 } as const;
export type Linkage = (typeof Linkage)[keyof typeof Linkage];

export interface Cluster {
	id: number;
	/** Indices into the dataset. */
	members: number[];
}

export interface Merge {
	a: number;
	b: number;
	into: number;
	sizeA: number;
	sizeB: number;
	/** Distance between the two clusters when they merged. */
	height: number;
}

export interface HierarchicalState {
	clusters: Cluster[];
	merges: Merge[];
	nextId: number;
}

export function initHierarchical(data: Dataset): HierarchicalState {
	return { clusters: data.map((_, i) => ({ id: i, members: [i] })), merges: [], nextId: data.length };
}

/** Summarizes all pairwise distances between two clusters into one number. */
function clusterDistance(a: Cluster, b: Cluster, data: Dataset, linkage: Linkage): number {
	let nearest = Infinity;
	let farthest = 0;
	let total = 0;
	for (const i of a.members) {
		for (const j of b.members) {
			const d = Math.hypot(data[i].x - data[j].x, data[i].y - data[j].y);
			nearest = Math.min(nearest, d);
			farthest = Math.max(farthest, d);
			total += d;
		}
	}
	if (linkage === Linkage.Single) return nearest;
	if (linkage === Linkage.Complete) return farthest;
	return total / (a.members.length * b.members.length);
}

/** Merges the two closest clusters. */
export function hierarchicalStep(state: HierarchicalState, data: Dataset, linkage: Linkage): HierarchicalState {
	const { clusters } = state;
	if (clusters.length < 2) return state;

	let [first, second, height] = [0, 1, Infinity];
	for (let i = 0; i < clusters.length; i++) {
		for (let j = i + 1; j < clusters.length; j++) {
			const d = clusterDistance(clusters[i], clusters[j], data, linkage);
			if (d < height) [first, second, height] = [i, j, d];
		}
	}

	const [a, b] = [clusters[first], clusters[second]];
	const merged: Cluster = { id: state.nextId, members: [...a.members, ...b.members] };
	return {
		clusters: clusters.flatMap((c, i) => (i === first ? [merged] : i === second ? [] : [c])),
		merges: [...state.merges, { a: a.id, b: b.id, into: merged.id, sizeA: a.members.length, sizeB: b.members.length, height }],
		nextId: state.nextId + 1,
	};
}

/** Cluster index per point. */
export function hierarchicalLabels(state: HierarchicalState, size: number): number[] {
	const labels = new Array<number>(size).fill(0);
	state.clusters.forEach((cluster, index) => {
		for (const member of cluster.members) labels[member] = index;
	});
	return labels;
}
