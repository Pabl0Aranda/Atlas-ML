import type { Dataset } from '../../types/ml';
import { createRng } from './random';

export interface Centroid {
	x: number;
	y: number;
}

export interface KMeansState {
	centroids: Centroid[];
	/** Cluster index per point. */
	assignments: number[];
	iteration: number;
	inertia: number;
	inertiaHistory: number[];
	converged: boolean;
}

export function nearestCentroid(centroids: Centroid[], x: number, y: number): number {
	let best = 0;
	let bestDistance = Infinity;
	centroids.forEach((c, i) => {
		const d = (c.x - x) ** 2 + (c.y - y) ** 2;
		if (d < bestDistance) {
			best = i;
			bestDistance = d;
		}
	});
	return best;
}

function assign(data: Dataset, centroids: Centroid[]) {
	const assignments = data.map((p) => nearestCentroid(centroids, p.x, p.y));
	const inertia = data.reduce((s, p, i) => {
		const c = centroids[assignments[i]];
		return s + (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
	}, 0);
	return { assignments, inertia };
}

/** k-means++ seeding: each new centroid is sampled proportionally to squared distance. */
export function initKMeans(data: Dataset, k: number, seed = 11): KMeansState {
	const rng = createRng(seed);
	const first = data[Math.floor(rng() * data.length)];
	const centroids: Centroid[] = [{ x: first.x, y: first.y }];
	while (centroids.length < Math.min(k, data.length)) {
		const weights = data.map((p) => {
			const c = centroids[nearestCentroid(centroids, p.x, p.y)];
			return (p.x - c.x) ** 2 + (p.y - c.y) ** 2;
		});
		let target = rng() * weights.reduce((s, w) => s + w, 0);
		let index = weights.findIndex((w) => (target -= w) <= 0);
		if (index < 0) index = data.length - 1;
		centroids.push({ x: data[index].x, y: data[index].y });
	}
	const { assignments, inertia } = assign(data, centroids);
	return { centroids, assignments, iteration: 0, inertia, inertiaHistory: [inertia], converged: false };
}

/** One Lloyd iteration: move each centroid to its cluster mean, then reassign. */
export function kMeansStep(state: KMeansState, data: Dataset): KMeansState {
	const centroids = state.centroids.map((c, i) => {
		const members = data.filter((_, j) => state.assignments[j] === i);
		if (!members.length) return c;
		return {
			x: members.reduce((s, p) => s + p.x, 0) / members.length,
			y: members.reduce((s, p) => s + p.y, 0) / members.length,
		};
	});
	const { assignments, inertia } = assign(data, centroids);
	return {
		centroids,
		assignments,
		iteration: state.iteration + 1,
		inertia,
		inertiaHistory: [...state.inertiaHistory, inertia],
		converged: assignments.every((a, i) => a === state.assignments[i]),
	};
}
