import type { Dataset } from '../../types/ml';
import { planeDistance, PlaneMetric } from './distance';
import { quantile } from './preprocessing';
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

/** What each algorithm minimizes: squared distance for k-means, plain distance for k-medians. */
function pointCost(metric: PlaneMetric, c: Centroid, x: number, y: number): number {
	const d = planeDistance(metric, c.x, c.y, x, y);
	return metric === PlaneMetric.Euclidean ? d * d : d;
}

export function nearestCentroid(centroids: Centroid[], x: number, y: number, metric: PlaneMetric = PlaneMetric.Euclidean): number {
	let best = 0;
	let bestDistance = Infinity;
	centroids.forEach((c, i) => {
		const d = pointCost(metric, c, x, y);
		if (d < bestDistance) {
			best = i;
			bestDistance = d;
		}
	});
	return best;
}

function assign(data: Dataset, centroids: Centroid[], metric: PlaneMetric) {
	const assignments = data.map((p) => nearestCentroid(centroids, p.x, p.y, metric));
	const inertia = data.reduce((s, p, i) => s + pointCost(metric, centroids[assignments[i]], p.x, p.y), 0);
	return { assignments, inertia };
}

/** k-means++ seeding: each new centroid is sampled proportionally to squared distance. */
export function initKMeans(data: Dataset, k: number, seed = 11, metric: PlaneMetric = PlaneMetric.Euclidean): KMeansState {
	const rng = createRng(seed);
	const first = data[Math.floor(rng() * data.length)];
	const centroids: Centroid[] = [{ x: first.x, y: first.y }];
	while (centroids.length < Math.min(k, data.length)) {
		const weights = data.map((p) => {
			return pointCost(metric, centroids[nearestCentroid(centroids, p.x, p.y, metric)], p.x, p.y);
		});
		let target = rng() * weights.reduce((s, w) => s + w, 0);
		let index = weights.findIndex((w) => (target -= w) <= 0);
		if (index < 0) index = data.length - 1;
		centroids.push({ x: data[index].x, y: data[index].y });
	}
	const { assignments, inertia } = assign(data, centroids, metric);
	return { centroids, assignments, iteration: 0, inertia, inertiaHistory: [inertia], converged: false };
}

/**
 * One Lloyd iteration: move each centroid to the center of its cluster, then reassign.
 * The mean minimizes squared Euclidean distance, but Manhattan distance needs the median.
 */
export function kMeansStep(state: KMeansState, data: Dataset, metric: PlaneMetric = PlaneMetric.Euclidean): KMeansState {
	const center = (values: number[]) =>
		metric === PlaneMetric.Euclidean ? values.reduce((s, v) => s + v, 0) / values.length : quantile(values, 0.5);
	const centroids = state.centroids.map((c, i) => {
		const members = data.filter((_, j) => state.assignments[j] === i);
		if (!members.length) return c;
		return { x: center(members.map((p) => p.x)), y: center(members.map((p) => p.y)) };
	});
	const { assignments, inertia } = assign(data, centroids, metric);
	return {
		centroids,
		assignments,
		iteration: state.iteration + 1,
		inertia,
		inertiaHistory: [...state.inertiaHistory, inertia],
		converged: assignments.every((a, i) => a === state.assignments[i]),
	};
}
